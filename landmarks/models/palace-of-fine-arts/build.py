"""Generate the landmark in metres. Local coordinates are across, along and up."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import manifold3d as md
import numpy as np

HERE = Path(__file__).resolve().parent
FACTS = json.loads((HERE / "facts.json").read_text())

def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def box(x0, x1, s0, s1, bottom, top):
    return md.Manifold.cube((x1 - x0, s1 - s0, top - bottom)).translate((x0, s0, bottom))


def section(outline):
    return md.CrossSection([outline], md.FillRule.EvenOdd)


def front_plate(outline, station, depth):
    return section(outline).extrude(depth).rotate((90, 0, 0)).translate((0, station + depth / 2, 0))


def loft(rings, cap=None):
    """Skin aligned rings, CCW about the sweep axis. Convex caps can use a fan."""
    n = len(rings[0])
    if cap is None:
        cap = [(0, i, i+1) for i in range(1, n-1)]
    faces = [(c, b, a) for a, b, c in cap]
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            faces.extend([(a, b, b + n), (a, b + n, a + n)])
    end = n * (len(rings) - 1)
    faces.extend((end+a, end+b, end+c) for a, b, c in cap)
    return md.Manifold(md.Mesh64(np.vstack(rings), np.array(faces, dtype=np.uint64)))


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


def mass(outline, low, high):
    return section(outline).extrude(high-low).translate((0, 0, low))


def drum(x, y, levels, sides=32):
    """Revolve horizontal elliptical sections (elevation, X radius, Y radius)."""
    angles = np.arange(sides) * math.tau / sides
    rings = [[(x + w*math.cos(t), y + d*math.sin(t), z) for t in angles]
             for z, w, d in levels]
    return loft(rings)


def arch_cut(x, bottom, spring, radius, station, depth):
    """Round-headed opening in a wall normal to the station axis."""
    arc = [(x+radius*math.cos(t), spring+radius*math.sin(t))
           for t in np.linspace(0, math.pi, 7)]
    return front_plate([(x-radius,bottom),(x+radius,bottom)]+arc, station, depth)


def finish(hard, curved):
    base = union(hard).as_original().simplify(0.018).calculate_normals(0, 0.5)
    round_parts = [s.calculate_normals(0, angle) for s, angle in curved]
    write_glb(union([base, *round_parts]).simplify(0.018))


def write_glb(solid):
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError("Solid failed its manifold check")
    mesh = solid.to_mesh64()
    basis = map_axes()
    props = mesh.vert_properties
    positions = (props[:, :3] @ basis.T).astype("<f4")
    normals = props[:, 3:6] @ basis.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    # Trimming a round member can introduce a small corner at a junction.
    # Split that corner when its interpolated normal points through the surface.
    triangles = mesh.tri_verts.copy()
    # A boolean intersection can leave sub-millimetre slivers. Collapse only
    # short mesh edges; all shading corners at an endpoint move together.
    points, owners = np.unique(positions, axis=0, return_inverse=True)
    spatial = owners[triangles]
    edges = np.sort(np.vstack((spatial[:, [0,1]], spatial[:, [1,2]], spatial[:, [2,0]])), axis=1)
    short = edges[np.linalg.norm(points[edges[:,0]]-points[edges[:,1]], axis=1) < 0.0012]
    parent = np.arange(len(points))
    for a, b in short:
        while parent[a] != a:
            a = parent[a]
        while parent[b] != b:
            b = parent[b]
        parent[max(a,b)] = min(a,b)
    while np.any(parent != parent[parent]):
        parent = parent[parent]
    positions = points[parent[owners]]
    corners = parent[spatial]
    triangles = triangles[np.all(corners != np.roll(corners, 1, axis=1), axis=1)]
    pa, pb, pc = positions[triangles].astype(np.float64).transpose(1, 0, 2)
    face_normals = np.cross(pb-pa, pc-pa)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)
    faces, corners = np.where(np.einsum("ij,ikj->ik", face_normals, normals[triangles]) < 0.05)
    if len(faces):
        positions = np.vstack((positions, positions[triangles[faces, corners]]))
        normals = np.vstack((normals, face_normals[faces]))
        triangles[faces, corners] = np.arange(len(faces)) + len(props)
    normals = normals.astype("<f4")
    # Preserve seams while sharing identical position/normal pairs.
    used, triangles = np.unique(triangles, return_inverse=True)
    triangles = triangles.reshape(-1,3)
    positions, normals = positions[used], normals[used]
    keys = np.hstack((positions, normals))
    _, first, remap = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    positions, normals = positions[first], normals[first]
    if len(positions) >= 65536:
        raise ValueError("Index range exceeded")
    indices = remap[triangles].astype("<u2").ravel()
    views, data = [], bytearray()
    for array, target in [(positions, 34962), (normals, 34962), (indices, 34963)]:
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=array.nbytes, target=target))
        data.extend(array.tobytes())
        data.extend(bytes(-len(data) % 4))
    linear = [((c/255+0.055)/1.055)**2.4 for c in (244, 241, 236)]
    model = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": FACTS["name"]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0}]}],
        "materials": [{"name": "body", "pbrMetallicRoughness": {
            "baseColorFactor": linear+[1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "accessors": [
            dict(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                 min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
            dict(bufferView=1, componentType=5126, count=len(normals), type="VEC3"),
            dict(bufferView=2, componentType=5123, count=len(indices), type="SCALAR")],
        "bufferViews": views, "buffers": [{"byteLength": len(data)}]}
    metadata = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    length = 28 + len(metadata) + len(data)
    glb = (struct.pack("<4sIII4s", b"glTF", 2, length, len(metadata), b"JSON") +
           metadata + struct.pack("<I4s", len(data), b"BIN\0") + data)
    if len(glb) > (1_500_000 if FACTS["kind"] == "bridge" else 400_000) or len(indices) // 3 > (25_000 if FACTS["kind"] == "bridge" else 6_000):
        raise ValueError(f"Mobile budget exceeded: {len(indices)//3} triangles, {len(glb)} bytes")
    (HERE / (FACTS["slug"] + ".glb")).write_bytes(glb)
    print(f"{len(indices)//3:,} triangles, {len(glb):,} bytes")


def build():
    plans=FACTS["plans"]
    hard,curved=[],[]
    for name in FACTS["ground_keys"]:
        outline=section(plans[name]["outline"])
        if name=="base":
            # The rotunda stands on one continuous terrace. Round both the
            # deep stair notches and their outer corners into broad landings.
            platform=outline.offset(2.4,md.JoinType.Round,12).offset(-2.4,md.JoinType.Round,12)
            platform=platform.offset(-1.9,md.JoinType.Round,12).offset(1.9,md.JoinType.Round,12)
        else:
            platform=outline.offset(1.7,md.JoinType.Round,8).offset(-1.7,md.JoinType.Round,8)
            platform=platform.offset(-1.2,md.JoinType.Round,8).offset(1.2,md.JoinType.Round,8)
        hard.append(platform.simplify(.58).extrude(.98))
    for i in range(8):
        outline=plans["pier"+str(i)]["outline"]
        x,y=plans["pier"+str(i)]["centre"]
        theta=math.atan2(y,x)
        pier=section(outline).offset(-.85,md.JoinType.Round,8).offset(.85,md.JoinType.Round,8)
        hard.append(pier.simplify(.35).extrude(3.1))
        for sign in (-1,1):
            u=x-sign*1.8*math.sin(theta);v=y+sign*1.8*math.cos(theta)
            hard.append(drum(u,v,[(2.9,1.55,1.55),
                                  (18.8,1.3,1.3),(20.2,2.1,2.1),(21.4,2.1,2.1)],5))
        hard.append(mass(outline,21.4,23))
    def ring(r,inside,low,high,n=16):
        return (md.CrossSection.circle(r,n)-md.CrossSection.circle(inside,n)).extrude(high-low).translate((0,0,low))
    # Wide round arches bridge the gaps between the paired supports.
    theta=math.atan2(*reversed(plans["pier0"]["centre"]))
    angles=[theta+math.radians(i*45+22.5+d) for i in range(8) for d in (-20.5,20.5)]
    def belt(r):
        return section([(r*math.cos(t),r*math.sin(t)) for t in angles])
    arcade=(belt(23.5)-belt(16.1)).extrude(7.5).translate((0,0,23))
    for i in range(8):
        cut=arch_cut(0,21.8,22.6,5.4,20,18)
        arcade-=cut.rotate((0,0,math.degrees(theta)-90+22.5+i*45))
    hard.extend([arcade,ring(24.384,16,30.1,31.5),ring(22.15,16.35,31.4,37.95)])
    hard.append(ring(22.7,16.6,37,38.45))
    hard.append(ring(19.6,15.5,38.25,40.7))
    # An open soffit follows the dome. The inner shell ends below its apex.
    radii=[(40.55,17.8),(42.55,16.75),(45.3,13.8),(47.5,9.75),
           (48.85,5.15),(49.3776,.7)]
    outer=drum(0,0,[(z,r,r) for z,r in radii],20)
    inner=drum(0,0,[(40.1,16.45,16.45)]+[(z-1.1,r-1.05,r-1.05) for z,r in radii[1:-1]]+[(48.2,.65,.65)],20)
    curved.append((outer-inner,24))
    paths={"north":[(-34.4,23),(-32.8,32),(-28.8,45),(-23.2,56),(-17.2,65),
                    (-8.8,73.8),(2,79),(16,83),(13,99),(11,110)],
           "south":[(-25.5,-32),(-20.3,-42.8),(-12.5,-52.8),(-5,-60.6),
                    (4.7,-67.6),(14,-72.9),(25.8,-76),(41.3,-73.6),(44.7,-89),(47,-99)]}
    for name,locations in paths.items():
        route=np.array(locations,dtype=float)
        # The beam follows the column centres. Each solid corner is wide
        # enough for the capitals instead of tapering to a polygonal sliver.
        beams=[]
        groups={0,3,6,7,9}
        for i,(a,b) in enumerate(zip(route,route[1:])):
            d=b-a
            length=np.linalg.norm(d)
            t=d/length
            n=np.array([-t[1],t[0]])
            beams.append(section([a-2.5*n-.75*t,b-2.5*n+.75*t,
                                  b+2.5*n+.75*t,a+2.5*n-.75*t]))
            intervals=round(length/6)
            for j in range(intervals):
                if j==0 and i in groups: continue
                u,v=a+d*j/intervals
                hard.append(drum(u,v,[(0,.95,.95),(14.9,.78,.78),
                                      (16.2,1.22,1.22)],5))
        hard.append(md.CrossSection.batch_boolean(beams,md.OpType.Add)
                    .extrude(3.98).translate((0,0,16.2)))
        for i in sorted(groups):
            centre=route[i]
            t=route[min(i+1,len(route)-1)]-route[max(0,i-1)]
            t=t/np.linalg.norm(t)
            n=np.array([-t[1],t[0]])
            for along in (-1.46,1.46):
                for across in (-1.38,1.38):
                    u,v=centre+t*along+n*across
                    hard.append(drum(u,v,[(0,.94,.94),(16.2,.84,.84)],4))
            outline=section([centre+t*i+n*j for i,j in
                             [(-2.7,-2.6),(2.7,-2.6),(2.7,2.6),(-2.7,2.6)]])
            hard.append(outline.extrude(5.136).translate((0,0,16.2)))
    finish(hard,curved)


if __name__ == "__main__":
    build()

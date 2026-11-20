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


def pedestal(x, s, levels):
    """Clipped-corner courses. Equal elevations produce horizontal setbacks."""
    rings = []
    for z, width, depth, chamfer in levels:
        w, d, c = width / 2, depth / 2, chamfer
        corners = [(-w+c, -d), (w-c, -d), (w, -d+c), (w, d-c),
                   (w-c, d), (-w+c, d), (-w, d-c), (-w, -d+c)]
        rings.append([(x + u, s + v, z) for u, v in corners])
    return loft(rings)


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


def bar(start, end, width, depth=None):
    """Rectangular member joining two points."""
    if depth is None:
        depth = width
    a, b = np.asarray(start), np.asarray(end)
    u = (b-a)/np.linalg.norm(b-a)
    side = np.cross(u, (0, 0, 1) if abs(u[2]) < 0.9 else (0, 1, 0))
    side /= np.linalg.norm(side)
    other = np.cross(u, side)
    matrix = np.column_stack((side, other, u, (a+b)/2))
    return md.Manifold.cube((width, depth, np.linalg.norm(b-a)), center=True).transform(matrix)


def tube(path, radius, sides=8):
    """Sweep a round cable along a station/elevation path at a fixed offset."""
    rings = []
    for i, (x, y, z) in enumerate(path):
        v = np.subtract(path[min(i+1, len(path)-1)], path[max(i-1, 0)])
        v = v/np.linalg.norm(v)
        across = np.cross(v, (0, 0, 1))
        if np.linalg.norm(across) < 1e-8:
            across = np.array([1., 0., 0.])
        across /= np.linalg.norm(across)
        normal = np.cross(v, across)
        angles = np.arange(sides)*math.tau/sides
        rings.append([np.array([x,y,z])+radius*(math.cos(t)*across+math.sin(t)*normal) for t in angles])
    return loft(rings)


def finish(hard, curved):
    base = union(hard).as_original().simplify(0.002).calculate_normals(0, 0.5)
    round_parts = [s.calculate_normals(0, angle) for s, angle in curved]
    write_glb(union([base, *round_parts]).simplify(0.002))


def write_glb(solid):
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError("Solid failed its manifold check")
    mesh = solid.to_mesh64()
    basis = map_axes()
    props = mesh.vert_properties
    positions = (props[:, :3] @ basis.T).astype("<f4")
    normals = props[:, 3:6] @ basis.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Trimming a round member can introduce a small corner at a junction.
    # Split that corner when its interpolated normal points through the surface.
    triangles = mesh.tri_verts.copy()
    # A boolean intersection can leave sub-millimetre slivers. Collapse only
    # short mesh edges; all shading corners at an endpoint move together.
    points, owners = np.unique(positions, axis=0, return_inverse=True)
    spatial = owners[triangles]
    edges = np.sort(np.vstack((spatial[:, [0,1]], spatial[:, [1,2]], spatial[:, [2,0]])), axis=1)
    short = edges[np.linalg.norm(points[edges[:,0]]-points[edges[:,1]], axis=1) < 0.004]
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
    guide=np.asarray(FACTS["skyway_guide"])
    # Trim the curved viaduct to 600 m measured along its centreline.
    sky=guide[guide[:,0]>=405]
    chain=np.r_[0,np.cumsum(np.linalg.norm(np.diff(sky,axis=0),axis=1))]
    end=np.array([np.interp(600,chain,sky[:,i]) for i in (0,1)])
    guide=np.vstack((guide[guide[:,0]<end[0]],end))
    def offset(y):
        return np.interp(y,guide[:,0],guide[:,1])-1.196
    def road(y):
        return 53-0.028*max(y,0)
    hard,curved=[],[]
    for sign in (-1,1):
        rings=[]
        for y in guide[:,0]:
            z=road(y)
            # Box girders widen from the soffit to the driving surface.
            profile=[(-13.94,-.9),(-13.94,.82),(-12.84,.82),(-12.84,0),
                     (12.84,0),(12.84,.82),(13.94,.82),(13.94,-.9),
                     (10.35,-4.4),(-10.35,-4.4)]
            rings.append([(offset(y)+sign*20.77+x,y,z+h) for x,h in profile])
        cap=[(0,1,2),(0,2,3),(0,3,4),(0,4,7),(4,5,6),(4,6,7),(0,7,8),(0,8,9)]
        hard.append(loft(rings,cap))
    hard.append(pedestal(0,0,[(-3,27,27,5),(3,25,25,4)]))
    for a in (-1,1):
        for b in (-1,1):
            profile=[(-1.45,-1.9),(1.45,-1.9),(1.45,1.9),(-1.45,1.9)]
            rings=[[(a*x+u,b*y+v,z) for u,v in profile] for z,x,y in
                   [(2.9,6.9,6.3),(45,4.25,5.2),(62,4.15,4.9),(153,2.15,2.15)]]
            hard.append(loft(rings))
    for z in (24,44,64,84,104,124,144):
        w=6.9-(z-3)/150*4.75
        height=3.7 if z in (24,84,144) else 2.0
        for sign in (-1,1):
            hard.append(box(-w,w,sign*w-1.1,sign*w+1.1,z,z+height))
            hard.append(box(sign*w-1.1,sign*w+1.1,-w,w,z,z+height))
    hard.append(pedestal(0,0,[(151,11,11,1.4),(155,11,11,1.4),
                              (155,8.7,8.7,1.1),(160,6.4,6.4,1)]))
    # Support both carriageways at the clipped ends of the displayed span.
    for y,direction in ((guide[0,0],1),(end[0],-1)):
        a,b=sorted((y,y+5*direction))
        for sign in (-1,1):
            x=offset(y)+sign*20.77
            top=min(road(a),road(b))-3.7
            hard.append(pedestal(x,(a+b)/2,[(0,9,b-a,1.2),(3,8,b-a,1.2),
                                            (top-6,7,b-a,1.0),(top,15,b-a,1.4)]))
    for y in [-180,385,520,665,810]:
        z=road(y)
        for sign in (-1,1):
            x=offset(y)+sign*20.77
            hard.append(pedestal(x,y,[(-2,11,12,1.5),(4,9,10,1.3),
                                       (z-10,7,8,1.2),(z-3.7,15,9,1.6)]))
        if y in (-180,385):
            hard.append(box(offset(y)-32,offset(y)+32,y-3.8,y+3.8,z-5,z-2))
    reach=20.77+13.94+1.35
    for sign in (-1,1):
        x=offset(385)+sign*reach
        hard.append(box(x-2,x+2,382,388,road(385)-2.2,road(385)+2.4))
        x=offset(-180)+sign*reach
        hard.append(box(x-1.9,x+1.9,-183,-177,road(-180)-8.6,road(-180)-5.8))
    hard.append(box(offset(-180)-reach,offset(-180)+reach,-182,-178,road(-180)-9.8,road(-180)-7.7))
    for y in np.linspace(-150,360,12):
        hard.append(box(offset(y)-11,offset(y)+11,y-0.9,y+0.9,road(y)-3.1,road(y)-1.6))
    # Stations run from the western turning saddle to the eastern anchoring.
    left,right=[],[]
    for y in np.r_[np.arange(-180,385,10),385]:
        t=abs(y)/(180 if y<0 else 385)
        x=3.2+(reach-3.2)*min(1,t/.79)
        end_z=road(-180)-7.2 if y<0 else road(385)+1.2
        z=(1-t)*156 + t*end_z - (22 if y<0 else 81)*t*(1-t)
        left.append((offset(y)-x,y,z))
        right.append((offset(y)+x,y,z))
    # The return is a horizontal half-ellipse. One swept tube makes the entire rope.
    loop=[(offset(-180)+reach*math.cos(t),-180-17*math.sin(t),road(-180)-7.2)
          for t in np.linspace(math.pi,0,17)[1:-1]]
    curved.append((tube(list(reversed(left))+loop+right,0.89,8),67))
    for path,sign in [(left,-1),(right,1)]:
        for x,y,z in path[1:-1]:
            if abs(y)<0.1 or z<road(y)+1.4:continue
            endpt=(offset(y)+sign*33.0,y,road(y)-0.45)
            curved.append((tube([endpt,(x,y,z)],0.55,6),73))
    finish(hard,curved)


if __name__ == "__main__":
    build()

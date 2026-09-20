# /// script
# requires-python = ">=3.11"
# dependencies = ["pyproj==3.7.2"]
# ///
"""Acquire mapped paddock footprints and author restrained historical landmarks.

Run uv run tools/build_landmarks.py. OSM geometry remains ODbL. Building heights,
facade details and tree heights are artistic estimates, not a measured survey.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.openstreetmap.org/api/0.6/map.json?bbox=-122.337,39.535,-122.324,39.546"


def main():
    origin = json.loads((ROOT / "godot/data/track.json").read_text())["origin"]
    aerial = json.loads((ROOT / "data/reference/ortho-measurements.json").read_text())
    with urllib.request.urlopen(URL, timeout=60) as response:
        osm = json.load(response)
    nodes = {row["id"]: row for row in osm["elements"] if row["type"] == "node"}
    transform = Transformer.from_crs("EPSG:4326", "EPSG:6339", always_xy=True)
    def local(node):
        x, y = transform.transform(node["lon"], node["lat"])
        return [round(x-origin["easting"],3),round(origin["northing"]-y,3)]
    buildings, fences, paving, posts, trees, snapshots = [], [], [], [], [], []
    for way in osm["elements"]:
        if way["type"] != "way": continue
        tags = way.get("tags",{})
        relevant = "building" in tags or tags.get("barrier") in ["fence","yes"] or tags.get("amenity")=="parking"
        if not relevant: continue
        points = [local(nodes[key]) for key in way["nodes"]]
        # Keep mapped facility objects, not long surrounding county boundary fences.
        if any(abs(p[0])>700 or p[1]<-600 or p[1]>1200 for p in points): continue
        item = {"osm_way":way["id"],"version":way["version"],"points":points,"tags":tags}
        snapshots.append({"id":way["id"],"version":way["version"],"nodes":[[key,nodes[key]["version"],nodes[key]["lon"],nodes[key]["lat"]] for key in way["nodes"]]})
        if "building" in tags:
            kind = tags["building"]
            item["kind"] = kind
            item["height_m"] = 4.0 if kind=="carport" else (7.0 if tags.get("name")=="Thunderhill Clubhouse" else 4.0)
            item["height_status"] = "Estimated massing, no mapped height tag or architectural survey"
            buildings.append(item)
        elif tags.get("amenity")=="parking":
            paving.append(item)
        elif tags.get("barrier")=="fence":
            item["height_m"] = 1.8
            if way["id"] == 896075617:
                item["kind"] = "pit_wall"
                item["height_m"] = 1.05
                item["status"] = "OSM tags this pit boundary as fence; white concrete appearance interpreted from 2026 rider video at 00:10; dimensions estimated"
            fences.append(item)
        elif all(-30<p[0]<12 and -200<p[1]<750 for p in points):
            # Video at 00:10 confirms a pale pit wall in this mapped corridor.
            item["kind"] = "pit_wall"
            item["height_m"] = 1.05
            item["status"] = "Mapped barrier position; concrete appearance from rider video; dimensions estimated"
            fences.append(item)
    for node in nodes.values():
        tags = node.get("tags",{})
        if tags.get("highway")=="marshal_post":
            p = local(node)
            if abs(p[0])<650 and -550<p[1]<1100:
                posts.append({"osm_node":node["id"],"p":p,"name":tags.get("name",""),"status":"Mapped position, estimated small shelter dimensions"})
        if tags.get("natural")=="tree":
            trees.append({"osm_node":node["id"],"p":local(node),"height_m":6.0,"radius_m":3.0,"status":"Mapped tree position; estimated crown and height"})
    # Crown centers personally inspected on the georeferenced July 2022 NAIP image.
    # Coordinates below are normalized from the 1391x1792 inspection view.
    crowns = [(692,664,4),(694,694,4),(694,708,3.5),(681,737,4),(694,760,3),(694,775,3),(694,788,3),(695,801,3),(695,824,3),(698,1090,3.5),(681,1129,4.5),(674,1142,4),(660,1180,3),(625,1075,3),(598,1322,3),(597,1343,3),(598,1362,3),(598,1382,4),(594,482,3),(619,500,3),(620,518,3.5),(635,547,3),(640,559,3),(618,460,2.5),(662,543,3)]
    extent = aerial["extent"]
    for x,y,radius in crowns:
        easting = extent["xmin"] + x/1391.0*(extent["xmax"]-extent["xmin"])
        northing = extent["ymax"] - y/1792.0*(extent["ymax"]-extent["ymin"])
        trees.append({"p":[round(easting-origin["easting"],3),round(origin["northing"]-northing,3)],"height_m":round(radius*1.7,2),"radius_m":radius,"source_image_fraction":[x/1391.0,y/1792.0],"status":"Visible aerial crown center, approximate manual interpretation; radius and height estimated"})
    content = {"schema_version":1,"buildings":buildings,"fences":fences,"paving":paving,"marshal_posts":posts,"trees":trees,"metadata":{"osm_url":URL,"osm_geometry_snapshot_sha256":hashlib.sha256(json.dumps(snapshots,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"osm_snapshots":snapshots,"aerial_catalog_url":aerial["catalog_url"],"aerial_sha256":aerial["image_sha256"],"aerial_acquisition":"2022-07-15","video_reference":"https://www.youtube.com/watch?v=yVjzZqYbKuM at 00:10, pale pit wall, trees and paddock building massing","license":"Mapped geometry and derived landmark database ODbL 1.0, OpenStreetMap contributors; original USDA/USGS aerial reference public domain; original rendering code MIT","osm_license_url":"https://www.openstreetmap.org/copyright","limitations":["OSM footprints are mapped observations, not surveyed architectural plans","Building heights, facade details, tree crown sizes and fence heights are estimated","Aerial tree evidence predates repave","No copyrighted video pixels are distributed as textures","Landmarks are visual geometry only, collision behavior not implemented"]}}
    assert buildings and paving and trees
    (ROOT/"godot/data/landmarks.json").write_text(json.dumps(content,separators=(",",":"))+"\n")
    print(json.dumps({"buildings":len(buildings),"fences":len(fences),"paving":len(paving),"marshal_posts":len(posts),"trees":len(trees),"pit_walls":[x["osm_way"] for x in fences if x.get("kind")=="pit_wall"]}))


if __name__=="__main__": main()

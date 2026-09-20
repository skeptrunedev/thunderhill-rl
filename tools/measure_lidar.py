# /// script
# requires-python = ">=3.11"
# dependencies = ["laspy[lazrs]==2.6.1", "numpy==2.4.3", "scipy==1.17.1", "pyproj==3.7.2", "matplotlib==3.10.8"]
# ///
"""Acquire original USGS lidar and measure selected road cross sections.

Run with uv run tools/measure_lidar.py. Outputs are historical observations,
not a post repave survey. USGS point clouds are public domain.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import laspy
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reference/lidar"


def acquire(product):
    target = OUT / product["downloadURL"].rsplit("/", 1)[1]
    if not target.exists():
        prefix = target.with_suffix(".part")
        initial_size = prefix.stat().st_size if prefix.exists() else 0
        ranges = [(start, min(start + 4_000_000, product["sizeInBytes"]) - 1) for start in range(initial_size, product["sizeInBytes"], 4_000_000)]
        def fetch_range(bounds):
            start, end = bounds
            chunk = target.with_suffix(f".range-{start}-{end}")
            if not chunk.exists() or chunk.stat().st_size != end - start + 1:
                subprocess.run(["curl", "--silent", "--show-error", "--fail", "--location", "--retry", "2", "--connect-timeout", "30", "--max-time", "120", "--range", f"{start}-{end}", "--output", str(chunk), product["downloadURL"]], check=True)
            if chunk.stat().st_size != end - start + 1:
                raise RuntimeError(f"Server did not honor requested byte range: {chunk}")
            return chunk
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            chunks = list(pool.map(fetch_range, ranges))
        with target.open("wb") as result:
            if initial_size:
                result.write(prefix.read_bytes())
            for chunk in chunks:
                result.write(chunk.read_bytes())
        print(f"Acquired {target.name}", flush=True)
    if target.stat().st_size != product["sizeInBytes"]:
        raise RuntimeError(f"Unexpected byte count: {target}")
    return target


def analyze(paths, summaries, reuse_corridor=False):
    geojson = json.loads((ROOT / "artifacts/reference/geometry/east-centerline.geojson").read_text())
    ll = np.array(geojson["features"][0]["geometry"]["coordinates"])
    with laspy.open(paths[0]) as reader:
        crs = reader.header.parse_crs()
    convert = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = convert.transform(ll[:, 0], ll[:, 1])
    route = np.column_stack([x, y])
    chainage = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    def point(s):
        s = np.asarray(s) % chainage[-1]
        return np.column_stack([np.interp(s, chainage, route[:, 0]), np.interp(s, chainage, route[:, 1])])
    dense = point(np.arange(0, chainage[-1], 2.0))
    route_tree = cKDTree(dense)
    if reuse_corridor:
        pts = np.load(OUT / "road-ground-points.npz")["points"]
    else:
        near = []
        for path in paths:
            print(f"Extracting road corridor {path.name}", flush=True)
            with laspy.open(path) as reader:
                for chunk in reader.chunk_iterator(1_000_000):
                    ground = (np.asarray(chunk.classification) == 2) & (np.asarray(chunk.withheld) == 0)
                    xy = np.column_stack([np.asarray(chunk.x)[ground], np.asarray(chunk.y)[ground]])
                    distance, _ = route_tree.query(xy, workers=4)
                    keep = distance <= 30
                    near.append(np.column_stack([xy[keep], np.asarray(chunk.z)[ground][keep], np.asarray(chunk.intensity)[ground][keep], np.asarray(chunk.point_source_id)[ground][keep]]))
        pts = np.concatenate(near)
        np.savez_compressed(OUT / "road-ground-points.npz", points=pts, columns=np.array(["x", "y", "z", "intensity", "point_source_id"]))
    tree = cKDTree(pts[:, :2])
    stations = list(np.arange(0, chainage[-1], 100.0))
    # Additional dense sampling around the mapped Cyclone branch.
    for lon, lat in [(-122.3290314,39.540432),(-122.329187,39.5407951),(-122.3292314,39.5410324),(-122.3294606,39.5410664),(-122.3298879,39.5413248)]:
        q = np.array(convert.transform(lon,lat))
        idx = np.argmin(np.linalg.norm(route-q,axis=1))
        stations.append(float(chainage[idx]))
    sections = []
    for station in sorted(set(stations)):
        center = point([station])[0]
        heading = point([station + 5])[0] - point([station - 5])[0]
        heading /= np.linalg.norm(heading)
        left = np.array([-heading[1],heading[0]])
        local = pts[tree.query_ball_point(center, 26)]
        delta = local[:, :2] - center
        t, n = delta @ heading, delta @ left
        strip = np.abs(t) <= 2
        core = strip & (np.abs(n) <= 3)
        if core.sum() < 30:
            continue
        A = np.column_stack([np.ones(core.sum()), t[core], n[core]])
        fit = np.linalg.lstsq(A,local[core,2],rcond=None)[0]
        residual = local[core,2] - A @ fit
        # Sensitivity to fitted core width is empirical, not a confidence interval.
        slopes=[]
        for half_width in [2,3,4]:
            mask=strip & (np.abs(n)<=half_width)
            aa=np.column_stack([np.ones(mask.sum()),t[mask],n[mask]])
            slopes.append(float(np.degrees(np.arctan(np.linalg.lstsq(aa,local[mask,2],rcond=None)[0][2]))))
        coords=Transformer.from_crs(crs,"EPSG:4326",always_xy=True).transform(*center)
        section={"station_m":float(station),"center_lon_lat":list(coords),"center_xy_m":center.tolist(),"heading_xy":heading.tolist(),"left_normal_xy":left.tolist(),"strip_length_m":4,"core_width_m":6,"core_point_count":int(core.sum()),"center_elevation_m":float(fit[0]),"along_slope_pct":float(fit[1]*100),"left_cross_slope_pct":float(fit[2]*100),"left_cross_slope_deg":float(np.degrees(np.arctan(fit[2]))),"plane_rmse_m":float(np.sqrt(np.mean(residual**2))),"cross_slope_deg_width_sensitivity_4_6_8m":slopes,"classification":2,"status":"Measured local strip; road edges and OSM center offset require orthophoto inspection."}
        edges=np.arange(-25,25.001,.5); bins=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=strip & (n>=lo) & (n<hi)
            if mask.sum(): bins.append({"offset_m":float((lo+hi)/2),"count":int(mask.sum()),"median_intensity":float(np.median(local[mask,3])),"median_height_detrended_m":float(np.median(local[mask,2]-fit[1]*t[mask]))})
        section["profile_bins"]=bins
        sections.append(section)
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout="constrained")
        axes[0].scatter(n[strip],local[strip,2]-fit[1]*t[strip],s=2,color="#275f8c",alpha=.6)
        axes[0].plot([-3,3],[fit[0]-3*fit[2],fit[0]+3*fit[2]],color="red")
        axes[0].set_ylabel("Height NAVD88 m, grade removed")
        axes[0].set_title(f"OSM chainage {station:.1f} m | local cross slope {section['left_cross_slope_deg']:.2f} degrees | 2023 lidar")
        axes[1].plot([b["offset_m"] for b in bins],[b["median_intensity"] for b in bins],".-")
        axes[1].set_ylabel("Median lidar intensity")
        axes[1].set_xlabel("Offset left of approximate OSM centerline, meters")
        for ax in axes:ax.grid(alpha=.2);ax.axvspan(-3,3,color="red",alpha=.05);ax.set_xlim(-25,25)
        fig.savefig(OUT / f"profile-{station:07.1f}.png",dpi=120);plt.close(fig)
    report={"status":"Historical raw lidar measurements, not an exact 2026 survey","source_year":2023,"ground_class":2,"corridor_point_count":len(pts),"crs":str(crs),"horizontal_and_vertical_units":"meters","tiles":summaries,"osm_license":"ODbL 1.0 for derived sample station geometry; source USGS points public domain","method":"Ordinary least squares elevation plane in 4m along by 6m across strip centered on approximate OSM route. Positive across slope rises left relative to route order. 4m and 8m core widths report sensitivity. No pavement boundaries assumed.","sections":sections}
    (OUT / "all-sections.json").write_text(json.dumps(report,indent=2)+"\n")
    compact = {k: v for k, v in report.items() if k != "sections"}
    compact["station_origin"] = {"osm_way_id": 28825115, "osm_way_version": 18, "first_node_id": 570035219, "lon_lat": ll[0].tolist(), "direction": "Increasing distance follows OSM node order, initially south", "chainage_definition": "Cumulative Euclidean distance in lidar projected CRS, not surveyed stationing"}
    compact["source_accuracy"] = {"url": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/CA_NorthCoastRanges_B23/USGS_CA_NorthCoastRanges_B23_Project_Report.pdf", "scope": "Entire collection, not independent Thunderhill control points", "tested_nonvegetated_vertical_rmse_cm": 3.63, "tested_nonvegetated_vertical_95pct_cm": 7.11, "coordinate_quantization_m": 0.01, "local_fit_rmse_is_not_absolute_accuracy": True}
    compact["limitations"] = ["2023 collection predates 2026 repave", "A road strip fit is not a validated whole pavement camber profile", "OSM centerline has visible lateral offsets relative to lidar reflectance band", "Intensity changes do not uniquely identify asphalt boundaries", "No present day exact road widths or curb dimensions are asserted", "Sensitivity across fitted strip widths is not a confidence interval"]
    compact["license"] = "Measurement database using OSM station geometry: ODbL 1.0, OpenStreetMap contributors, separate from repository MIT code. Original USGS point clouds public domain."
    compact["osm_license_url"] = "https://www.openstreetmap.org/copyright"
    compact["sections"] = [{k:v for k,v in row.items() if k != "profile_bins"} for row in sections]
    (ROOT / "data/reference/lidar-measurements.json").write_text(json.dumps(compact,indent=2)+"\n")
    print(f"Measured {len(sections)} profiles",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-corridor", action="store_true", help="Explicitly reuse the previously generated local corridor for plot and measurement iteration")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/reference/geometry-sources.json").read_text())
    products = [p for q in manifest["catalog_queries"] for p in q["products"] if p["format"] == "LAZ" and "NorthCoastRanges_B23" in p["title"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(acquire, products))
    summaries = []
    for path in paths:
        data = laspy.read(path)
        classes, counts = np.unique(data.classification, return_counts=True)
        summary = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "point_count": len(data.points), "crs": str(data.header.parse_crs()), "scales": data.header.scales.tolist(), "bounds_min": data.header.mins.tolist(), "bounds_max": data.header.maxs.tolist(), "classification_counts": dict(zip(map(str, classes.tolist()), counts.tolist())), "intensity_percentiles": np.percentile(data.intensity, [0,25,50,75,100]).tolist(), "dimensions": list(data.point_format.dimension_names)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (OUT / "tile-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    analyze(paths, summaries, reuse_corridor=args.reuse_corridor)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Patch Rai1ly/zerorf_test to run COLMAP train -> held-out test metrics -> one mesh.

Run from the repository root:
    python zerorf_colmap_eval_mesh_patch.py

Targets the GitHub main revision inspected on 2026-10-25. Refuses to change
unexpected source code. Backups are written as <path>.before_colmap_eval_mesh.bak.
Does NOT modify your model/data files or push to GitHub.
"""

from pathlib import Path
import shutil
import sys


def replace_once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one original code block, found {count}. "
                           "Source differs from the checked GitHub version; no files changed.")
    return src.replace(old, new, 1)


def patch_opt(src: str) -> str:
    return replace_once(
        src,
        "    parser.add_argument('--rectify_convention', action='store_true', help='if rectify the convention')\n",
        "    parser.add_argument('--rectify_convention', action='store_true', help='if rectify the convention')\n"
        "    parser.add_argument('--test-every', type=int, default=10,\n"
        "                        help='for COLMAP, hold out every Nth image for testing')\n"
        "    parser.add_argument('--mesh-resolution', type=int, default=256,\n"
        "                        help='COLMAP final mesh voxel resolution (e.g. 256 or 384)')\n",
        "opt.py parser options",
    )


def patch_colmap_dataset(src: str) -> str:
    before, marker, colmap = src.partition("class ColmapDataset(Dataset):")
    if not marker:
        raise RuntimeError("ColmapDataset class not found")
    before = replace_once(before, "import os\n", "import os\nimport re\n", "COLMAP natural filename sorting import")

    colmap = replace_once(
        colmap,
        "    def __init__(self, root_dir: str, split: str, world_scale: float = 1.0, rgba: bool = True):\n",
        "    def __init__(self, root_dir: str, split: str, world_scale: float = 1.0, rgba: bool = True, test_every: int = 10, load_depth: bool = False):\n",
        "ColmapDataset.__init__",
    )
    colmap = replace_once(
        colmap,
        "        self.downscale = 3.57\n",
        "        self.downscale = 3.57\n"
        "        self.test_every = int(test_every)\n"
        "        if self.test_every < 2:\n"
        "            raise ValueError('test_every must be at least 2')\n",
        "ColmapDataset.downscale (must remain 3.57)",
    )
    colmap = replace_once(
        colmap,
        "        imkeys = np.array(sorted(imdata.keys()))\n",
        "        # Natural filename order: image2.png before image10.png.\n"
        "        def natural_image_key(k):\n"
        "            name = os.path.basename(imdata[k].name)\n"
        "            return tuple((0, int(t)) if t.isdigit() else (1, t.lower())\n"
        "                         for t in re.split(r'(\\d+)', name))\n"
        "        imkeys = np.array(sorted(imdata.keys(), key=natural_image_key))\n",
        "COLMAP filename ordering",
    )
    block_start = "        # make split\n"
    block_end = "        # read images\n"
    if colmap.count(block_start) != 1 or colmap.count(block_end) != 1:
        raise RuntimeError("COLMAP train/val split block not found; no files changed")
    left, rest = colmap.split(block_start, 1)
    _, right = rest.split(block_end, 1)
    replacement = '''        # Every test_every-th VALID image in filename order is held out.
        # Zero based indices 9, 19,... when test_every=10.
        all_ids = np.arange(len(img_paths))
        test_ids = all_ids[self.test_every - 1::self.test_every]
        if len(test_ids) == 0:
            raise ValueError(f'COLMAP found {len(all_ids)} valid images, too few for test_every={self.test_every}')
        train_ids = np.setdiff1d(all_ids, test_ids)
        if self.split == 'train':
            chosen_ids = train_ids
        elif self.split in ('val', 'test'):
            chosen_ids = test_ids
        else:
            raise ValueError(f'Unknown COLMAP split: {self.split}')

        self.poses = self.poses[chosen_ids]
        self.intrinsics = self.intrinsics[chosen_ids]
        img_paths = img_paths[chosen_ids]
        if mask_paths is not None:
            mask_paths = mask_paths[chosen_ids]
        self.image_names = [os.path.basename(p) for p in img_paths]
        print(f'[COLMAP] {self.split}: {len(img_paths)} images, filenames: {self.image_names}')

        # Depth is OPTIONAL. Only used if --enable_dense_depth is explicitly set.
        self.dense_depth_info = None
        if self.split == 'train' and load_depth:
            depth_folder = os.path.join(self.root_dir, 'depth')
            if not os.path.isdir(depth_folder):
                raise FileNotFoundError(f'Depth requested but directory is missing: {depth_folder}')
            if os.path.isdir(depth_folder):
                depth_images = []
                for img_path in tqdm.tqdm(img_paths, desc='Loading depth data'):
                    f = os.path.join(depth_folder, os.path.splitext(os.path.basename(img_path))[0] + '.png')
                    dep = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                    if dep is None:
                        raise FileNotFoundError(f'Cannot read depth file: {f}')
                    if dep.ndim == 3:
                        dep = dep[..., 0]
                    dep = cv2.resize(dep, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                    depth_images.append((dep.astype(np.float32) / 255.0)[..., None])
                self.dense_depth_info = torch.from_numpy(np.stack(depth_images, axis=0))

        # read images
'''
    colmap = left + replacement + right
    colmap = replace_once(
        colmap,
        "            cond_depth=np.array(self.dense_depth_info, np.float32),\n",
        "            cond_depth=None if self.dense_depth_info is None else np.array(self.dense_depth_info, np.float32),\n",
        "COLMAP optional depth in __getitem__",
    )
    return before + marker + colmap


def patch_zerorf(src: str) -> str:
    src = replace_once(
        src,
        '''        dataset = ColmapDataset(f"{args.data_dir}", split='train', world_scale=world_scale, rgba=True)
        val = ColmapDataset(f"{args.data_dir}", split='val', world_scale=world_scale)
        test = ColmapDataset(f"{args.data_dir}", split='val', world_scale=world_scale)
''',
        '''        dataset = ColmapDataset(f"{args.data_dir}", split='train', world_scale=world_scale,
                                rgba=True, test_every=args.test_every,
                                load_depth=args.enable_dense_depth)
        test = ColmapDataset(f"{args.data_dir}", split='test', world_scale=world_scale,
                             test_every=args.test_every)
''',
        "zerorf.py COLMAP dataset construction",
    )
    src = replace_once(
        src,
        "        selected_idxs = [i for i in range(16)]\n",
        '''        # Number of train views cannot exceed the held-in training pool.
        selected_idxs = list(range(min(args.n_views, len(entry['cond_imgs']))))
        print(f'[COLMAP] use {len(selected_idxs)} training views (requested {args.n_views})')
        args.n_views = len(selected_idxs)
''',
        "zerorf.py COLMAP selected_idxs",
    )
    src = replace_once(
        src,
        '''    entry = val[0]
    val_entry = dict(
        test_imgs=torch.tensor(entry['cond_imgs'][:args.n_val][None]).float().to(device),
        test_poses=torch.tensor(entry['cond_poses'][:args.n_val])[None].float().to(device),
        test_intrinsics=torch.tensor(entry['cond_intrinsics'][:args.n_val])[None].float().to(device),
        scene_id=[0],
        scene_name=[args.proj_name]
    )
''',
        '''    # No validation / model selection on held-out COLMAP test images.
    if args.dataset != 'colmap':
        entry = val[0]
        val_entry = dict(
            test_imgs=torch.tensor(entry['cond_imgs'][:args.n_val][None]).float().to(device),
            test_poses=torch.tensor(entry['cond_poses'][:args.n_val])[None].float().to(device),
            test_intrinsics=torch.tensor(entry['cond_intrinsics'][:args.n_val])[None].float().to(device),
            scene_id=[0],
            scene_name=[args.proj_name]
        )
''',
        "zerorf.py validation input",
    )
    # Add split manifest immediately after test_entry creation.
    src = replace_once(
        src,
        '''        scene_name=[args.proj_name]
    )
else:
    data_entry = dict(
''',
        '''        scene_name=[args.proj_name]
    )
    if args.dataset == 'colmap':
        with open('split.json', 'w', encoding='utf-8') as f:
            json.dump({
                'data_dir': args.data_dir,
                'test_every': args.test_every,
                'downscale': dataset.downscale,
                'train_images': [dataset.image_names[i] for i in selected_idxs],
                'test_images': test.image_names,
            }, f, ensure_ascii=False, indent=2)
else:
    data_entry = dict(
''',
        "zerorf.py split manifest",
    )
    src = replace_once(
        src,
        "    if j % args.val_iter == args.val_iter - 1:\n",
        '''    # COLMAP: no 60-frame video, no repeated val/test, and no test-driven best selection.
    # Train for the requested number of iterations, then test and mesh once.
    if args.dataset == 'colmap':
        if j == args.n_iters - 1:
            torch.save(nerf.state_dict(), 'nerf-zerorf.pt')
            nerf.eval()
            scene_code = nerf.cache[0]['param']['code_'][None].to(device)
            density_bitfield = nerf.cache[0]['param']['density_bitfield'][None].to(device)

            with torch.no_grad():
                test_metrics, _ = nerf.eval_and_viz(
                    test_entry, nerf.decoder, scene_code, density_bitfield,
                    viz_dir=None, cfg=nerf.test_cfg
                )
            print('[COLMAP] Final held-out test metrics:', test_metrics)
            wandb.log(dict(test=test_metrics))
            with open('test_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2)

            # Export ONE scene mesh, not one mesh per held-out camera.
            nerf.decoder.visualize_mesh = True
            nerf.decoder.mesh_resolution = args.mesh_resolution
            os.makedirs('mesh', exist_ok=True)
            with torch.no_grad():
                nerf.decoder.visualize(scene_code, [args.proj_name], 'mesh')
            print(f'[COLMAP] Mesh saved: mesh/{args.proj_name}.glb')
        continue

    if j % args.val_iter == args.val_iter - 1:
''',
        "zerorf.py final COLMAP test & mesh branch",
    )
    return src


def patch_decoder(src: str) -> str:
    old = "        basis = torch.linspace(-1, 1, 500, device=code.device, dtype=code.dtype)\n"
    new = "        mesh_resolution = int(getattr(self, 'mesh_resolution', 256))\n" \
          "        basis = torch.linspace(-1, 1, mesh_resolution, device=code.device, dtype=code.dtype)\n"
    if src.count(old) != 2:
        raise RuntimeError("Expected two mesh extraction methods (TensorialDecoder and FreqFactorizedDecoder)")
    src = src.replace(old, new)
    if src.count("        verts = verts / 250 - 1\n") != 2:
        raise RuntimeError("Expected two fixed 500-grid vertex scaling expressions")
    src = src.replace("        verts = verts / 250 - 1\n",
                      "        verts = verts / (mesh_resolution - 1) * 2 - 1\n")
    return src


def main():
    root = Path.cwd()
    paths = {
        'opt.py': patch_opt,
        'zerorf.py': patch_zerorf,
        'lib/datasets/custom.py': patch_colmap_dataset,
        'lib/models/zerorf/decoders.py': patch_decoder,
    }
    results = {}
    for relative, fn in paths.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find {relative}. Run this script in the ZeroRF repository root.")
        original = path.read_text(encoding='utf-8')
        updated = fn(original)
        compile(updated, str(path), 'exec')
        results[path] = (original, updated)

    for path, (original, updated) in results.items():
        backup = path.with_name(path.name + '.before_colmap_eval_mesh.bak')
        if backup.exists():
            raise RuntimeError(f"Backup already exists: {backup}; check previous edits before re-running.")
    for path, (original, updated) in results.items():
        backup = path.with_name(path.name + '.before_colmap_eval_mesh.bak')
        shutil.copy2(path, backup)
        path.write_text(updated, encoding='utf-8')
        print('PATCHED', path.relative_to(root), '(backup:', backup.name + ')')
    print('Done. Inspect with: git diff -- opt.py zerorf.py lib/datasets/custom.py lib/models/zerorf/decoders.py')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(1)

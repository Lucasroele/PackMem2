from multiprocessing import Process, Queue
from queue import Empty

from tqdm import tqdm

from packmem2.core import param as p
import packmem2.packmem2
import packmem2.concatenate
import packmem2.analysis
import sys


def _launch_worker(
    topo, traj, lipid, start, end, paramFile, radiiFile, indexFile,
    output_dir, outputname, dist_suppl_Z, protein, pdbout, wrapped,
    progress_queue,
) -> None:
    packmem2.packmem2._progress_queue = progress_queue
    packmem2.packmem2.launch(
        topo, traj, lipid, start, end, paramFile, radiiFile, indexFile,
        output_dir, outputname, dist_suppl_Z, protein, pdbout, wrapped,
    )

def launch(
    cores: int,
    topo: str,
    traj: str,
    lipid: str,
    start: int,
    end: int,
    paramFile: str,
    radiiFile: str,
    indexFile: str | None,
    output_dir: str,
    outputname: str,
    dist_suppl_Z: float,
    protein: bool,
    pdbout: bool,
    limx: int | float,
    limy: float,
    precision: int,
    nb_block: int,
    wrapped: bool,
) -> None:
    """
    Launch the different python scripts for PackMem2
    """
    ######## PackMem2 #######
    # Prepare how much frames are going to be analysed per core
    step = int(max(end / cores, 1))
    # Create a list of the minima / maxima frame of each interval
    intervals = list(range(start, end + 1, step))
    processes = []
    progress_queue = Queue()

    # Launch PackMem2 on multiple processes
    for i in range(len(intervals) - 1):
        # Select starting and ending frame (limits of the interval)
        start_i = intervals[i]
        end_i = intervals[i + 1] - 1
        # If this is the last interval
        if i == (len(intervals) - 2):
            end_i = end
        p = Process(
            target=_launch_worker,
            args=(
                topo,
                traj,
                lipid,
                start_i,
                end_i,
                paramFile,
                radiiFile,
                indexFile,
                output_dir,
                outputname,
                dist_suppl_Z,
                protein,
                pdbout,
                wrapped,
                progress_queue,
            ),
        )
        p.start()
        processes.append(p)

    with tqdm(total=end - start + 1, desc="Frames", unit="frame") as progress:
        while any(p.is_alive() for p in processes):
            try:
                progress.update(progress_queue.get(timeout=0.1))
            except Empty:
                pass
        for p in processes:
            p.join()
        while True:
            try:
                progress.update(progress_queue.get_nowait())
            except Empty:
                break

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        sys.exit(f"{len(failed)} worker process(es) failed; see the errors above.")

    ######## Concatenate #######
    packmem2.concatenate.launch(output_dir, output_dir, outputname, start, end, protein, False)

    ######## Analysis #######
    # Then launch packing defect analysis
    packmem2.analysis.launch(output_dir, output_dir, outputname, protein, nb_block, limx, limy, precision)


def main() -> None:
    args = p.get_args_launch_packmem2()

    launch(
        args.cores,
        args.topo,
        args.traj,
        args.lipid,
        args.start,
        args.end,
        args.paramFile,
        args.radiiFile,
        args.indexFile,
        args.output_dir,
        args.outputname,
        args.dist_suppl_Z,
        args.protein,
        args.pdbout,
        args.limx,
        args.limy,
        args.precision,
        args.nb_block,
        args.wrapped,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python

if __name__ != '__main__':
    raise Exception("Do not import me!")

from pathlib import Path
import logging


from multiprocessing import Pool
from tqdm.auto import tqdm

import mothseg
from mothseg import parser
from mothseg import utils


def proceed_check(yes: bool) -> bool:
    return yes or input("Do you want to proceed? [y/N] ").lower() == "y"


def main(args):
    config = utils.read_config(args.config)
    images = utils.find_images(args.folder)

    logging.info(f"Found {len(images):,d} images in {args.folder}")
    first10 = '\n'.join(map(str, images[:10]))
    logging.debug(f"Here are the first 10: \n{first10}")

    if not proceed_check(args.yes):
        logging.info("Execution aborted by the user!")
        return -1

    output = utils.check_output(args, use_timestamp=args.use_timestamp)

    if output is None:
        logging.info("No output folder selected, exiting script.")
        return -2
    else:
        if not proceed_check(args.yes):
            logging.info("Execution aborted by the user!")
            return -1

    if args.n_jobs > 1 or args.n_jobs == -1:
        N = args.n_jobs if args.n_jobs > 0 else None
        pool = Pool(N)
        mapper = pool.imap
        logging.info(f"Initialized worker with {pool._processes} processes...")
    else:
        pool = None
        mapper = map

    logging.info("Starting processing...")
    with tqdm(total=len(images)) as bar:

        writer = mothseg.OutputWriter(output, config=args.config)
        plotter = mothseg.Plotter(output, plot_interm=args.plot_interm)
        worker = mothseg.Worker(config, plotter,
                        progress_callback=None, #bar.set_description,
                        raise_on_error=args.raise_on_error)

        for impath, stats, e in mapper(worker, images):
            bar.set_description(f"Processing {Path(impath).name}")
            if e is not None:
                writer.log_fail(impath, e)
                continue
            assert stats is not None
            writer(impath, stats)
            bar.update(1)

    if pool is not None:
        logging.info("Waiting for all processes to finish...")
        pool.close()
        pool.join()

    logging.info("All done!")


exit(main(parser.parse_args()))

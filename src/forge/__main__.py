from forge.cli import main

# Guarded: multiprocessing on Windows re-imports this module in every worker.
if __name__ == "__main__":
    main()

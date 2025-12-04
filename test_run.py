"""Quick test with 5 generations"""
import sys
sys.path.insert(0, '.')

# Patch the constant before importing
import main
main.NUM_GENERATIONS = 5

# Run the program
if __name__ == "__main__":
    main.main()

# main.py
import argparse
from enroll import enroll_face
from verify import verify_face

def main():
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument("-e", "--enroll", type=str, help="Enroll a new user. Provide the user's name.")
    parser.add_argument("-v", "--verify", action="store_true", help="Verify an existing user.")
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Set the threshold for face recognition.")
    
    args = parser.parse_args()
    
    if args.enroll:
        enroll_face(args.enroll)
    elif args.verify:
        verify_face(threshold=args.threshold)
    else:
        print("Please provide a valid option. Use -h for help.")

if __name__ == "__main__":
    main()

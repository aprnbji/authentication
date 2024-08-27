# authentication

**enroll()**
    save face encodings
    save face landmarks
**verify()**
    use saved encodings and landmarks to verify face. loop through saved entities and see if anything matches.
    if a match is found, return the name of the person along with the message "verified". it also checks for
    liveness of the face for further verification to prevent spoofing.
**main()**
    calls the verify() first to verify. if face is not recognized, it returns the message not recognized and 
    asks if you want your face to be enrolled. if yes, it calls the enroll() and then the verify(), otherwise 
    exits the program.

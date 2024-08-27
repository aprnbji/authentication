# authentication

**enroll()**

- saves face encodings
- saves face landmarks

**verify()**

- uses saved encodings and landmarks to verify face. 
- loops through saved entities and see if anything matches.
- if a match is found, returns the name of the person along with the message "verified"(it also checks for
    liveness of the face for further verification to prevent spoofing), otherwise returns "not recognized"
- if it is verified but the depth is 0m, it returns "not authenticated"

**main()**

- calls the verify() first to verify. 
- if face is not recognized, it returns the message not recognized and asks if you want your face to be enrolled. 
if yes, it calls the enroll() and then the verify(), otherwise exits the program.



import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import cv2
import numpy as np
import sys
import threading
import time

class JarvisAssistant:
    def __init__(self):
        # Initialize Voice Engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            sys.exit(1)

        # Initialize Speech Recognizer
        self.r = sr.Recognizer()
        
        # UI State
        self.ui_text = "INITIALIZING..."
        self.running = True
        self.lock = threading.Lock()

    def speak(self, text):
        """Speaks the text using the TTS engine."""
        print(f"Jarvis: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def update_ui_text(self, text):
        """Thread-safe UI text update."""
        with self.lock:
            self.ui_text = text

    def listen(self):
        """Listens to microphone input and returns the recognized text."""
        command = ""
        try:
            with sr.Microphone() as source:
                self.update_ui_text("LISTENING...")
                print("Listening...")
                self.r.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.r.listen(source, timeout=5, phrase_time_limit=5)
                    self.update_ui_text("PROCESSING...")
                    command = self.r.recognize_google(audio)
                    print(f"You said: {command}")
                except sr.WaitTimeoutError:
                    pass # Just loop back
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
        except OSError as e:
             print(f"Microphone error: {e}")
             self.speak("Microphone not found or inaccessible.")
             self.running = False
        
        return command.lower()

    def jarvis_ui_loop(self):
        """Runs the OpenCV UI in a separate loop/thread or main loop."""
        # Note: cv2.imshow must often run in the main thread on some OSs.
        # We will try running it here.
        while self.running:
            img = np.zeros((500, 800, 3), np.uint8)
            
            with self.lock:
                current_text = self.ui_text

            # Draw "Available" Indicator or Animation Placeholder
            cv2.circle(img, (400, 150), 50, (0, 255, 255), 2)
            
            # Text
            cv2.putText(img, "JARVIS AI", (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Dynamic Status
            cv2.putText(img, current_text, (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(img, "Press 'q' to quit", (10, 480),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("JARVIS", img)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                self.running = False
        
        cv2.destroyAllWindows()

    def process_command(self, command):
        """Executes actions based on the command."""
        if not command:
            return

        self.update_ui_text(f"CMD: {command.upper()}")

        if "time" in command:
            time_now = datetime.datetime.now().strftime("%I:%M %p")
            self.update_ui_text(f"TIME: {time_now}")
            self.speak(f"Current time is {time_now}")

        elif "open youtube" in command:
            self.speak("Opening YouTube")
            self.update_ui_text("OPENING YOUTUBE")
            webbrowser.open("https://youtube.com")

        elif "open google" in command:
            self.speak("Opening Google")
            self.update_ui_text("OPENING GOOGLE")
            webbrowser.open("https://google.com")

        elif "play music" in command:
            self.speak("Playing music")
            self.update_ui_text("PLAYING MUSIC")
            webbrowser.open("https://open.spotify.com/")

        elif "who are you" in command:
            self.speak("I am Jarvis, your AI assistant")
            self.update_ui_text("I AM JARVIS")

        elif "bye" in command or "exit" in command or "quit" in command:
            self.speak("Goodbye boss")
            self.update_ui_text("SYSTEM SHUTDOWN")
            self.running = False

        else:
            self.speak("Sorry, I didn't understand")
            self.update_ui_text("UNKNOWN COMMAND")

    def run(self):
        """Main execution method."""
        print("Starting Jarvis...")
        self.speak("Jarvis Activated")
        
        # Start UI in a separate thread implies complexity with cv2.imshow main thread requirements.
        # Instead, we'll try a non-blocking listen approach or just alternate.
        # Since listen() blocks, we need threads.
        # Common issue: cv2.imshow usually needs to be in the main thread.
        # So we will put the logic in a thread and UI in main.
        
        logic_thread = threading.Thread(target=self.logic_loop)
        logic_thread.daemon = True
        logic_thread.start()
        
        self.jarvis_ui_loop() # This blocks main thread until quit

    def logic_loop(self):
        while self.running:
            command = self.listen()
            if command:
                self.process_command(command)
            time.sleep(0.1)

if __name__ == "__main__":
    app = JarvisAssistant()
    app.run()

import cv2
import mediapipe as mp
import numpy as np
import random
import math

class LightningEffect:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Changed to 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def get_lightning_points(self, start_point, end_point, segments=10, glitter_strength=25):
        """
        Generates a list of points creating a zigzag path between start and end.
        """
        points = [start_point]
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return [start_point, end_point]

        nx = -dy / distance
        ny = dx / distance

        for i in range(1, segments):
            t = i / segments
            base_x = start_point[0] + dx * t
            base_y = start_point[1] + dy * t
            
            # Dynamic jitter based on distance (farther = more jitter)
            dynamic_strength = glitter_strength * (distance / 400.0) 
            dynamic_strength = max(10, min(dynamic_strength, 50)) # Clamp
            
            offset = random.uniform(-dynamic_strength, dynamic_strength)
            jx = int(base_x + nx * offset)
            jy = int(base_y + ny * offset)
            points.append((jx, jy))
            
        points.append(end_point)
        return points

    def draw_glowing_line(self, img, start, end, color=(255, 255, 0), thickness=2):
        """
        Helper to draw a single lightning bolt with glow.
        """
        # We'll return the mask of this specific bolt to add to the main accumulator
        h, w = img.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        points = self.get_lightning_points(start, end, segments=12)
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        
        # Glow
        cv2.polylines(canvas, [pts], False, color, thickness + 12)
        canvas = cv2.GaussianBlur(canvas, (35, 35), 0)
        
        # Core
        cv2.polylines(canvas, [pts], False, (255, 255, 255), thickness)
        
        return canvas

    def get_coords(self, landmark, width, height):
        return int(landmark.x * width), int(landmark.y * height)

    def run(self):
        cap = cv2.VideoCapture(0)
        # Set Resolution to HD (1280x720) directly on the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create a named window and set it to Fullscreen
        window_name = "Chandru's Chidori"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("Starting Enhanced Lightning...")
        print("Mode 1: ONE HAND -> Connect Index & Pinky")
        print("Mode 2: TWO HANDS -> Connect matching fingertips (Cage Effect)")
        
        while True:
            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Accumulator for all lightning effects this frame
            final_effect = np.zeros_like(img)

            if results.multi_hand_landmarks:
                hands = results.multi_hand_landmarks
                
                # CASE 1: Single Hand Detected
                if len(hands) == 1:
                    lm = hands[0].landmark
                    p1 = self.get_coords(lm[8], w, h)  # Index
                    p2 = self.get_coords(lm[20], w, h) # Pinky
                    
                    bolt = self.draw_glowing_line(img, p1, p2, color=(255, 255, 0), thickness=3)
                    final_effect = cv2.add(final_effect, bolt)
                
                # CASE 2: Two Hands Detected
                elif len(hands) == 2:
                    # Sort hands by x-coordinate to consistently identify left/right in frame
                    # This prevents crossing lines if hands swap positions 
                    # (though sorting by handedness info is better, simple x sort works for visuals)
                    hands.sort(key=lambda h_lm: h_lm.landmark[0].x)
                    
                    h1 = hands[0].landmark
                    h2 = hands[1].landmark
                    
                    # Connect matching fingers [Thumb, Index, Middle, Ring, Pinky]
                    # IDs: 4, 8, 12, 16, 20
                    finger_ids = [4, 8, 12, 16, 20]
                    
                    for fid in finger_ids:
                        p1 = self.get_coords(h1[fid], w, h)
                        p2 = self.get_coords(h2[fid], w, h)
                        
                        # Draw bolt
                        bolt = self.draw_glowing_line(img, p1, p2, color=(255, 200, 0), thickness=2)
                        final_effect = cv2.add(final_effect, bolt)

            # Combine original image with the accumulated effect
            img = cv2.add(img, final_effect)
            
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LightningEffect()
    app.run()

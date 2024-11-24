import pygame
import random
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load the pre-trained model and configuration
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Labels
labels_dict = {0: 'P', 1: 'B', 2: 'C', 3: 'H', 4: 'L', 5: 'O', 6: 'R', 7: 'V'}
questions = list(labels_dict.values())

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("ASL BUDDY")
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
NEON_GREEN = (0, 255, 100)
NEON_BLUE = (0, 255, 255)
NEON_PURPLE = (150, 50, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Scoring
score = 0  # Initialize the score

# Function to display text
def display_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

# Function to draw futuristic hand landmarks
def draw_futuristic_landmarks(frame, hand_landmarks, connections):
    # Draw glowing dots on landmarks
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 8, RED, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 12, NEON_GREEN, 2, cv2.LINE_AA)

    # Draw connections between landmarks
    for connection in connections:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        x1, y1 = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
        x2, y2 = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), NEON_PURPLE, 2, cv2.LINE_AA)

# Function to detect hand gesture and return the predicted letter
def detect_sign(target_letter):
    global score  # Access the score variable
    cap = cv2.VideoCapture(0)
    detected_label = None
    correct_time = None  # To track when the correct answer was detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        box_color = RED  # Default to red if no match
        if results.multi_hand_landmarks:
            # Process only one hand (e.g., the one with the lowest x coordinate)
            primary_hand = min(results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].x)

            data_aux = []
            x_ = []
            y_ = []

            # Draw futuristic landmarks
            draw_futuristic_landmarks(frame, primary_hand, mp_hands.HAND_CONNECTIONS)

            for i in range(len(primary_hand.landmark)):  # 21 landmarks
                x = primary_hand.landmark[i].x
                y = primary_hand.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize coordinates
            min_x = min(x_)
            min_y = min(y_)

            for i in range(len(primary_hand.landmark)):  # 21 landmarks
                data_aux.append(primary_hand.landmark[i].x - min_x)
                data_aux.append(primary_hand.landmark[i].y - min_y)

            # Ensure the feature count matches the model's requirement
            if len(data_aux) != 42:
                # Skip this frame if the feature count is incorrect
                print(f"Skipping frame: Feature count {len(data_aux)} != 42")
                continue

            # Predict using the model
            prediction = model.predict([np.asarray(data_aux)])
            detected_label = labels_dict[int(prediction[0])]

            # Determine box color
            box_color = GREEN if detected_label == target_letter else RED

            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

        # If correct gesture is detected, start 2-second timer
        if detected_label == target_letter:
            if correct_time is None:
                correct_time = time.time()  # Start the timer
                score += 10  # Increment the score by 10
            cv2.putText(frame, "Correct!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, NEON_GREEN, 3, cv2.LINE_AA)

        # Check if 2 seconds have passed since the correct answer
        if correct_time and time.time() - correct_time > 2:
            break

        # Display detected label
        cv2.putText(frame, f"Detected: {detected_label or '...'}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2, cv2.LINE_AA)
        cv2.imshow("Sign Detection", frame)

        # Press 'q' to exit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main game loop
running = True
game_active = False
question = None
feedback = ""
questions_asked = []  # To track already asked questions

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if not game_active and 300 <= mouse_x <= 500 and 250 <= mouse_y <= 300:
                game_active = True
                question = random.choice(questions)
                questions_asked.append(question)

            if game_active and 300 <= mouse_x <= 500 and 450 <= mouse_y <= 500:
                detect_sign(question)

                # Move to next question
                if len(questions_asked) < len(questions):
                    remaining_questions = [q for q in questions if q not in questions_asked]
                    question = random.choice(remaining_questions)
                    questions_asked.append(question)
                else:
                    feedback = "Game Over! You've completed all questions."
                    game_active = False

    if not game_active:
        # Main menu
        display_text("ASL BUDDY", font, BLACK, 250, 100)
        pygame.draw.rect(screen, BLUE, (300, 250, 200, 50))
        display_text("Start Game", small_font, WHITE, 330, 260)
    else:
        # Game screen
        display_text(f"Question: Show '{question}'", small_font, BLACK, 275, 100)
        display_text(f"Score: {score}", small_font, BLACK, 10, 10)  # Display the score
        pygame.draw.rect(screen, BLUE, (300, 450, 200, 50))
        display_text("Submit", small_font, WHITE, 350, 460)

        if feedback:
            display_text(feedback, small_font, BLACK, 200, 300)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
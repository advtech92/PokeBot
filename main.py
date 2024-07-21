import numpy as np
import torch
import tkinter as tk
from game_control import press_key, bring_window_to_front
from screen_capture import capture_screen
from dqn import DQN, preprocess_image, choose_action, train_step


# Initialize previous state and position
prev_state = None
stagnation_counter = 0
max_stagnation_steps = 2  # Maximum number of steps allowed without movement
in_new_room = False


def calculate_reward(state, prev_state):
    global stagnation_counter, in_new_room
    reward = 0

    if prev_state is None:
        return reward

    # Reward for moving out of the starting room
    if detect_movement(state, prev_state):
        reward += 10
        stagnation_counter = 0  # Reset counter if movement is detected
        in_new_room = True
        print("Movement detected, reward +10")
    else:
        stagnation_counter += 1  # Increment counter if no movement
        print("No movement detected, stagnation counter increased to", stagnation_counter)

    # Reward for interacting with objects
    if detect_interaction(state):
        reward += 5
        print("Interaction detected, reward +5")

    # Punish for hitting walls or not making progress
    if stagnation_counter > max_stagnation_steps:
        reward -= 10
        stagnation_counter = 0  # Reset counter to avoid repeated punishment
        print("Stagnation detected, penalty -10")

    # Punish for leaving the new room too soon
    if in_new_room and not detect_movement(state, prev_state):
        reward -= 5
        in_new_room = False
        print("Left new room too soon, penalty -5")

    # Small penalty for each move to encourage efficient actions
    reward -= 1
    print("Small move penalty -1")

    print("Reward calculated:", reward)
    return reward


def detect_movement(state, prev_state, threshold=0.01):
    try:
        # Check if NumPy is available
        if not hasattr(np, 'abs'):
            raise RuntimeError("NumPy is not available")

        # Calculate the absolute difference between the current and previous state
        diff = np.abs(state - prev_state)

        # Calculate the proportion of pixels that have changed significantly
        num_changed_pixels = np.sum(diff > threshold)
        total_pixels = np.prod(state.shape)

        # Consider movement significant if more than a certain percentage of pixels have changed
        movement = (num_changed_pixels / total_pixels) > threshold
        return movement

    except Exception as e:
        print(f"Error in detect_movement: {e}")
        raise


def detect_interaction(state):
    interaction = False
    return interaction


def is_done(state):
    done = False
    return done


def update_action_display(action, reward):
    actions = ['Up', 'Down', 'Left', 'Right', 'Interact']
    action_text = f"Current Action: {actions[action]}"
    reward_text = f"Reward/Penalty: {reward}"
    action_label.config(text=action_text)
    reward_label.config(text=reward_text)


def main():
    bring_window_to_front()
    net = DQN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    global prev_state

    for episode in range(1000):
        state = preprocess_image(capture_screen())
        prev_state = state
        done = False
        while not done:
            action = choose_action(net, state)
            if action == 0:
                press_key('up')
            elif action == 1:
                press_key('down')
            elif action == 2:
                press_key('left')
            elif action == 3:
                press_key('right')
            elif action == 4:
                press_key('a')  # Assuming 'a' is the key to interact/select

            next_state = preprocess_image(capture_screen())
            reward = calculate_reward(next_state, prev_state)
            done = is_done(next_state)
            train_step(net, optimizer, criterion, state, action, reward, next_state, done)
            prev_state = next_state
            state = next_state
            update_action_display(action, reward)
        print(f"Episode {episode} completed.")


if __name__ == "__main__":
    # Create a small window to display the bot's actions and rewards
    root = tk.Tk()
    root.title("Bot Action Display")
    action_label = tk.Label(root, text="Current Action: None", font=("Helvetica", 16))
    action_label.pack(pady=20)
    reward_label = tk.Label(root, text="Reward/Penalty: None", font=("Helvetica", 16))
    reward_label.pack(pady=20)

    # Run the main loop in a separate thread
    import threading
    main_thread = threading.Thread(target=main)
    main_thread.start()

    root.mainloop()

import numpy as np
import torch
from game_control import press_key, bring_window_to_front, load_save_state, save_state
from screen_capture import capture_screen
from dqn import DQN, preprocess_image, choose_action, train_step
import os
import tkinter as tk
from PIL import Image, ImageTk

# Initialize previous state and position
prev_state = None
stagnation_counter = 0
max_stagnation_steps = 2  # Maximum number of steps allowed without movement
in_new_room = False
dialogue_counter = 0  # Counter to track dialogue interactions
low_reward_threshold = -50  # Threshold for resetting the state
movement_cooldown = 5  # Steps to wait after significant movement
movement_cooldown_counter = 0  # Counter to track cooldown
select_day_counter = 0  # Counter for selecting the day of the week
town_map_counter = 0  # Counter for handling the town map

model_path = "dqn_model.pth"  # Path to save/load the model

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995

def calculate_reward(state, prev_state):
    global stagnation_counter, in_new_room, dialogue_counter, movement_cooldown_counter, select_day_counter, town_map_counter
    reward = 0

    if prev_state is None:
        return reward

    # Cooldown logic
    if movement_cooldown_counter > 0:
        movement_cooldown_counter -= 1
        return reward

    # Reward for moving out of the starting room
    if detect_movement(state, prev_state):
        reward += 10
        stagnation_counter = 0  # Reset counter if movement is detected
        in_new_room = True
        dialogue_counter = 0  # Reset dialogue counter
        town_map_counter = 0  # Reset town map counter
        movement_cooldown_counter = movement_cooldown  # Apply cooldown
    else:
        stagnation_counter += 1  # Increment counter if no movement
    
    # Reward for interacting with objects
    if detect_interaction(state):
        reward += 5

    # Punish for hitting walls or not making progress
    if stagnation_counter > max_stagnation_steps:
        reward -= 10
        stagnation_counter = 0  # Reset counter to avoid repeated punishment
    
    # Punish for leaving the new room too soon
    if in_new_room and not detect_movement(state, prev_state):
        reward -= 5
        in_new_room = False
    
    # Small penalty for each move to encourage efficient actions
    reward -= 1

    # Check for dialogue situations and interact
    if detect_dialogue(state):
        dialogue_counter += 1
        if select_day_counter < 7:  # Select day of the week (Monday to Sunday)
            press_key('right')  # Press 'Right' to move through days
            select_day_counter += 1
            reward += 1  # Small reward for interacting with dialogue
        elif town_map_counter < 5:  # Handle the town map scenario
            press_key('b')  # Press 'B' to back out of the town map
            town_map_counter += 1
            reward += 1  # Small reward for interacting with dialogue
        elif dialogue_counter < 5:  # Only press 'A' a few times to advance dialogue
            press_key('a')  # Press 'A' to advance dialogue
            reward += 1  # Small reward for interacting with dialogue

    return reward

def detect_movement(state, prev_state, threshold=0.01):
    try:
        # Ensure the tensors are converted to NumPy arrays
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state
        prev_state_np = prev_state.cpu().numpy() if torch.is_tensor(prev_state) else prev_state

        # Calculate the absolute difference between the current and previous state
        diff = np.abs(state_np - prev_state_np)

        # Calculate the proportion of pixels that have changed significantly
        num_changed_pixels = np.sum(diff > threshold)
        total_pixels = np.prod(state_np.shape)

        # Consider movement significant if more than a certain percentage of pixels have changed
        movement = (num_changed_pixels / total_pixels) > threshold
        return movement

    except Exception as e:
        print(f"Error in detect_movement: {e}")
        raise

def detect_interaction(state):
    # Implement logic to detect interaction with objects
    interaction = False
    # Placeholder: you can improve this by checking specific patterns in the state
    return interaction

def detect_dialogue(state):
    # Implement logic to detect dialogue situations
    # Placeholder: you can improve this by checking specific patterns in the state
    dialogue = False
    # Example: check for specific colors or text areas typical in dialogue
    # This requires more detailed analysis of game screen and patterns
    return dialogue

def is_done(state):
    # Implement logic to determine when an episode is done
    # Placeholder: could be based on reaching certain points in the game
    return False

def update_visualization(state, image_label):
    # Convert the state tensor to a NumPy array and ensure it's in the correct format
    state_np = state.cpu().numpy().squeeze() if torch.is_tensor(state) else state
    if state_np.ndim == 2:  # If the state is grayscale
        state_np = np.stack([state_np] * 3, axis=-1)  # Convert to RGB by stacking the same array thrice
    elif state_np.shape[0] == 1:  # If the state has a single channel
        state_np = np.concatenate([state_np] * 3, axis=0)  # Convert to RGB
    state_np = (state_np * 255).astype(np.uint8).transpose(1, 2, 0)  # Normalize and transpose to (H, W, C)

    # Convert the state to an image
    image = Image.fromarray(state_np)
    image = ImageTk.PhotoImage(image)
    image_label.config(image=image)
    image_label.image = image

def run_episode(episode, image_label):
    global total_reward, prev_state, epsilon_start
    print(f"Starting episode {episode}")
    load_save_state()  # Load the save state at the beginning of each episode
    state = preprocess_image(capture_screen())
    prev_state = state
    done = False
    episode_reward = 0
    select_day_counter = 0  # Reset day selection counter for each episode
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))  # Decay epsilon

    while not done:
        q_values = net(state).detach().numpy().flatten()
        action = choose_action(net, state, epsilon)
        print(f"Chosen action: {action}")
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
        episode_reward += reward
        done = is_done(next_state)
        train_step(net, optimizer, criterion, state, action, reward, next_state, done)
        prev_state = next_state
        state = next_state

        # Update the visualization
        update_visualization(next_state, image_label)
        root.update_idletasks()
        root.update()

        # Reset state if rewards get too low and not in dialogue
        if episode_reward < low_reward_threshold and not detect_dialogue(state):
            load_save_state()  # Load the save state
            print("Reward too low, resetting state")
            break

    total_reward += episode_reward
    save_state()  # Save state after episode

    # Save the model after each episode
    torch.save(net.state_dict(), model_path)
    print(f"Episode {episode} completed with reward {episode_reward}.")

def main():
    global net, optimizer, criterion, root

    bring_window_to_front()
    net = DQN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Load the model if it exists
    if os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path))
        net.eval()
        print("Model loaded successfully.")

    root = tk.Tk()
    root.title("Real-time Game Visualization")
    image_label = tk.Label(root)
    image_label.pack()

    # Run the main loop in a separate thread
    def start_episodes():
        for episode in range(1000):
            run_episode(episode, image_label)

    import threading
    main_thread = threading.Thread(target=start_episodes)
    main_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()

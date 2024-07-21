from game_control import press_key, bring_window_to_front
from screen_capture import capture_screen
from dqn import DQN, preprocess_image, choose_action, train_step

import torch


def main():
    bring_window_to_front()
    net = DQN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for episode in range(1000):
        state = preprocess_image(capture_screen())
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
            next_state = preprocess_image(capture_screen())
            reward = 1  # Placeholder reward
            done = False  # Implement game-specific logic to determine if done
            train_step(net, optimizer, criterion, state, action, reward, next_state, done)
            state = next_state
        print(f'Episode {episode} completed.')


if __name__ == "__main__":
    main()

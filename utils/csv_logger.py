import csv
import os


class CSVLogger:
    
    def __init__(self, filename):
        self.filename = filename

        # Step 1: Initialize the CSV with new headers
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['time_step', 'agent_x', 'agent_y', 'block_x', 'block_y', 'block_theta'])

    # Step 2: Append a row of data
    def log_data(self, time_step, agent_pos, block_pos):
        agent_x, agent_y = agent_pos
        block_x, block_y, block_theta = block_pos
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time_step, agent_x, agent_y, block_x, block_y, block_theta])
            
class ActionCSVLogger:
    
    def __init__(self, filename):
        self.filename = filename

        # Step 1: Initialize the CSV with new headers
        if not os.path.exists(filename):
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['time_step', 'action_x', 'action_y'])

    # Step 2: Append a row of data
    def log_data(self, time_step, actions):
        action_x, action_y = actions
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time_step, action_x, action_y])

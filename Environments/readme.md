# **Environments**

## **Overview**
The `Environments` folder contains all components related to environment definitions and their implementations. These environments serve as the foundation for running reinforcement learning (RL) experiments and are structured to allow easy customization and scalability.

---

## **Structure**
### **1. BaseEnvironment.py**
- **Purpose**: Provides a base class for defining environments. 
  - It includes fundamental methods that any RL environment should implement.
  - Methods such as `reset`, `isterminal`, `step`, `setaction`, `getreward`, and `getstate` are defined as abstract methods to be overridden by specific implementations.

- **Key Methods**:
  1. `reset()`: Resets the environment to its initial state.
  2. `isterminal()`: Checks if the current state is terminal.
  3. `step(action)`: Performs an action and updates the state.
  4. `setaction(action)`: Sets the current action to be executed.
  5. `getreward()`: Returns the reward for the current state and action.
  6. `getstate()`: Returns the current state of the environment.

---

### **2. example_CustomEnvironment**
- **Location**: `Environments/example_CustomEnvironment/CustomEnvironment.py`
- **Purpose**: Demonstrates how to create a custom environment by inheriting from `BaseEnvironment`.
  - This folder contains an example implementation of an environment for quick reference or prototyping.

- **Example Features**:
  - Overrides all required methods of `BaseEnvironment`.
  - Implements specific logic for state transitions, reward calculation, and terminal state detection.
  - Can serve as a template for creating new environments.

---

### **3. VectorizedEnvironment**
- **Location**: `Environments/VectorizedEnvironment/`
- **Purpose**: Placeholder for implementing vectorized environments.
  - Vectorized environments enable the simultaneous execution of multiple environments to improve training efficiency.
  - Future updates will provide utilities and examples for creating and managing such environments.

---

## **How to Use**

### **1. Create a New Environment**
- Use `BaseEnvironment` as the parent class.
- Override the abstract methods to define the specific behavior of your environment.
- Example:
  ```python
  from BaseEnvironment import BaseEnvironment

  class MyCustomEnvironment(BaseEnvironment):
      def __init__(self):
          super().__init__()
          # Initialization code here

      def reset(self):
          # Reset logic here
          return initial_state

      def step(self, action):
          # Step logic here
          return new_state, reward, done

      # Implement other methods as needed

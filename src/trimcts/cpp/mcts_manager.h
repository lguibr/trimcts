
#ifndef TRIMCTS_CPP_MCTS_MANAGER_H
#define TRIMCTS_CPP_MCTS_MANAGER_H

#pragma once

#include <memory>              // For std::unique_ptr
#include <pybind11/pybind11.h> // Include pybind11 for py::object
#include <cstdio>              // For fprintf in destructor error

// Forward declaration of Node to avoid circular include if possible
// If Node methods are needed here, include mcts.h instead.
namespace trimcts
{
  class Node;
}

namespace py = pybind11;

namespace trimcts
{

  class MctsTreeManager
  {
  public:
    // Constructor takes ownership of the root node
    explicit MctsTreeManager(std::unique_ptr<Node> root);

    // Destructor will automatically delete the managed root node via unique_ptr
    ~MctsTreeManager() = default;

    // --- Rule of 5/0 ---
    // Disable copy constructor and assignment
    MctsTreeManager(const MctsTreeManager &) = delete;
    MctsTreeManager &operator=(const MctsTreeManager &) = delete;
    // Enable move constructor and assignment (transfers ownership of root_)
    MctsTreeManager(MctsTreeManager &&) = default;
    MctsTreeManager &operator=(MctsTreeManager &&) = default;

    // Method to get the root node (non-owning pointer)
    Node *get_root() const;

    // Method to take ownership of the root (used for subtree promotion)
    std::unique_ptr<Node> release_root();

    // Method to set a new root (takes ownership)
    void set_root(std::unique_ptr<Node> new_root);

  private:
    std::unique_ptr<Node> root_;
  };

  // Destructor function for the py::capsule
  // IMPORTANT: This function MUST have C linkage to be correctly used by PyCapsule_New.
  extern "C"
  {
    inline void capsule_destructor(PyObject *capsule)
    {
      // Name must match the one used when creating the capsule in bindings.cpp
      // Use static_cast for type safety.
      MctsTreeManager *manager = static_cast<MctsTreeManager *>(PyCapsule_GetPointer(capsule, "MctsTreeManager"));
      if (manager)
      {                 // Check if pointer is not null
        delete manager; // Delete the MctsTreeManager object
      }
      else if (PyErr_Occurred())
      {
        // An error occurred during PyCapsule_GetPointer (e.g., wrong name)
        // Pybind11 might handle this, but logging could be useful.
        // Avoid throwing exceptions from destructors.
        fprintf(stderr, "Error: Failed to retrieve MctsTreeManager pointer from capsule in destructor.\n");
        PyErr_Clear(); // Clear the Python error state
      }
      else
      {
        // Pointer was null, but no Python error occurred (e.g., capsule created with nullptr)
        fprintf(stderr, "Warning: MctsTreeManager capsule destructor called with null pointer.\n");
      }
    }
  } // extern "C"

} // namespace trimcts

#endif // TRIMCTS_CPP_MCTS_MANAGER_H
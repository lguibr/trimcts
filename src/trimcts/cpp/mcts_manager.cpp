
#include "mcts_manager.h"
#include "mcts.h"  // Include full Node definition
#include <utility> // For std::move

namespace trimcts
{

  MctsTreeManager::MctsTreeManager(std::unique_ptr<Node> root) : root_(std::move(root)) {}

  Node *MctsTreeManager::get_root() const
  {
    return root_.get();
  }

  std::unique_ptr<Node> MctsTreeManager::release_root()
  {
    return std::move(root_);
  }

  void MctsTreeManager::set_root(std::unique_ptr<Node> new_root)
  {
    root_ = std::move(new_root);
  }

} // namespace trimcts
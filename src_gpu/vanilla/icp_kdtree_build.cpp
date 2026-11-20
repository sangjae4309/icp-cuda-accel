#include <iostream>
#include <array>
using namespace std;

template <size_t K>
class KDTree {
private:
    struct Node {
        array<double, K> point;
        Node* left;
        Node* right;

        Node(const array<double, K>& pt)
            : point(pt), left(nullptr), right(nullptr) {}
    };

    Node* root;

    // Insert a point recursively
    Node* insertRecursive(Node* node, const array<double, K>& point, int depth) {
        if (node == nullptr)
            return new Node(point);

        // Determine splitting axis: X -> Y -> Z -> X ...
        int cd = depth % K;

        if (point[cd] < node->point[cd])
            node->left = insertRecursive(node->left, point, depth + 1);
        else
            node->right = insertRecursive(node->right, point, depth + 1);

        return node;
    }

    // Search for an exact point
    bool searchRecursive(Node* node, const array<double, K>& point, int depth) const {
        if (node == nullptr)
            return false;

        if (node->point == point)
            return true;

        int cd = depth % K;

        if (point[cd] < node->point[cd])
            return searchRecursive(node->left, point, depth + 1);
        else
            return searchRecursive(node->right, point, depth + 1);
    }

    // Print the tree
    void printRecursive(Node* node, int depth) const {
        if (node == nullptr)
            return;

        for (int i = 0; i < depth; i++)
            cout << "  ";

        cout << "(";
        for (size_t i = 0; i < K; i++) {
            cout << node->point[i];
            if (i < K - 1)
                cout << ", ";
        }
        cout << ")" << endl;

        printRecursive(node->left, depth + 1);
        printRecursive(node->right, depth + 1);
    }

    // Free allocated memory
    void freeTree(Node* node) {
        if (node == nullptr)
            return;

        freeTree(node->left);
        freeTree(node->right);
        delete node;
    }

public:
    KDTree() : root(nullptr) {}

    ~KDTree() {
        freeTree(root);
    }

    // Prevent accidental shallow copying of owning pointers
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;

    void insert(const array<double, K>& point) {
        root = insertRecursive(root, point, 0);
    }

    bool search(const array<double, K>& point) const {
        return searchRecursive(root, point, 0);
    }

    void print() const {
        printRecursive(root, 0);
    }
};
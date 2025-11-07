# Contributing to Breast Cancer Classification Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top right corner of the repository

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

3. **Set Up Development Environment**
   ```bash
   conda env create -f environment.yml
   conda activate breast_cancer
   ```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Example:
```python
def extract_features(dataloader, model, device='cpu'):
    """
    Extract features from images using a pretrained model.
    
    Args:
        dataloader: PyTorch DataLoader with images
        model: Feature extraction model
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        tuple: (features, labels) as numpy arrays
    """
    # Implementation here
```

### Commit Messages

Use clear and descriptive commit messages:
- `feat: Add new ensemble model architecture`
- `fix: Correct confusion matrix calculation`
- `docs: Update README with new results`
- `refactor: Simplify data loading pipeline`
- `test: Add unit tests for metrics calculation`

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Test your changes thoroughly
   - Update documentation if needed

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Description of your changes"
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear description of changes

## 🧪 Testing

Before submitting a pull request:

1. **Test Your Code**
   ```bash
   # Test training script
   python scripts/model1_alexnet_vgg_svm.py --test-mode
   
   # Test visualization generation
   python scripts/generate_model1_visualizations.py
   ```

2. **Check for Errors**
   - Ensure no runtime errors
   - Verify outputs are correct
   - Check that visualizations render properly

## 📊 Adding New Models

If you're adding a new model:

1. **Create Model Script**
   - Place in `scripts/` directory
   - Follow naming convention: `modelX_description.py`

2. **Add Model Architecture**
   - Define in `src/training/` if reusable
   - Include docstrings and comments

3. **Generate Results**
   - Save metrics to `models/modelX/metrics.json`
   - Save visualizations to `results/model X/`
   - Create README in model directory

4. **Update Documentation**
   - Add to main README.md
   - Include performance metrics
   - Add usage examples

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Exact steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, GPU info
6. **Error Messages**: Full error traceback if applicable

**Template:**
```markdown
### Bug Description
[Clear description]

### Steps to Reproduce
1. Step 1
2. Step 2
3. ...

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Environment
- Python: 3.11.9
- PyTorch: 2.0.0
- OS: Windows 11 / Ubuntu 22.04
- GPU: NVIDIA RTX 3080

### Error Message
```
[Paste full error traceback]
```
```

## 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

1. **Clear Use Case**: Why this enhancement is needed
2. **Proposed Solution**: How it could be implemented
3. **Alternatives**: Other approaches you've considered
4. **Impact**: How this affects existing functionality

## 📂 Project Structure Guidelines

When adding new files:

- **Scripts**: Training/evaluation scripts → `scripts/`
- **Models**: Reusable model architectures → `src/training/`
- **Utils**: Helper functions → `src/utils/`
- **Data**: Data processing → `src/data/`
- **Evaluation**: Metrics and visualization → `src/evaluation/`

## ✅ Code Review Process

All submissions require review. Reviewers will check:

- Code quality and style
- Documentation completeness
- Test coverage
- Performance implications
- Compatibility with existing code

## 🎯 Priority Areas

Current areas where contributions are especially welcome:

1. **Model Improvements**
   - New architectures
   - Hyperparameter optimization
   - Ensemble methods

2. **Data Augmentation**
   - Advanced augmentation techniques
   - Domain-specific transformations

3. **Visualization**
   - Interactive plots
   - Additional metrics
   - Comparison tools

4. **Documentation**
   - Tutorial notebooks
   - API documentation
   - Usage examples

5. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

## 📞 Questions?

If you have questions:
- Open an issue with the "question" label
- Check existing issues and discussions
- Review the main README.md

## 🙏 Thank You!

Your contributions help make this project better for everyone!

---

**Note**: By contributing, you agree that your contributions will be licensed under the MIT License.

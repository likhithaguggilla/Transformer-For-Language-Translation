# TensorBoard Text Logging Guide

## What You'll See in TensorBoard

After running training with the updated code, you'll be able to view translation examples directly in TensorBoard!

### How to Access:

1. **Start TensorBoard:**
   ```bash
   tensorboard --logdir=runs
   ```

2. **Open in browser:**
   ```
   http://localhost:6006
   ```

3. **Navigate to the TEXT tab** (in addition to SCALARS)

---

## TensorBoard Tabs You'll Have:

### 1. SCALARS Tab
- `train_loss` - Training loss per batch
- `validation_loss` - Validation loss per epoch

### 2. TEXT Tab (NEW!)
- `validation_examples` - Table showing:
  - Source text (English)
  - Expected translation (French)
  - Predicted translation (Model output)

---

## Example of What You'll See:

### TEXT Tab - validation_examples:

| Source | Expected | Predicted |
|--------|----------|-----------|
| Hello, how are you? | Bonjour, comment allez-vous? | Bonjour, comment vas-tu? |
| I love machine learning. | J'adore l'apprentissage automatique. | J'aime l'apprentissage machine. |

The table updates after each epoch, so you can track how translations improve over time!

---

## Console Output (Terminal):

You'll also see this in your terminal during validation:

```
================================================================================
Running Validation for Epoch 00...
================================================================================
--------------------------------------------------------------------------------
VALIDATION EXAMPLE 1:
  SOURCE:    Hello, how are you?
  EXPECTED:  Bonjour, comment allez-vous?
  PREDICTED: Bonjour, comment vas-tu?
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
VALIDATION EXAMPLE 2:
  SOURCE:    I love machine learning.
  EXPECTED:  J'adore l'apprentissage automatique.
  PREDICTED: J'aime l'apprentissage machine.
--------------------------------------------------------------------------------

Validation Loss: 3.2145
```

---

## Benefits:

✅ **Visual comparison** - See source, expected, and predicted side-by-side
✅ **Track progress** - Watch translations improve over epochs
✅ **Easy debugging** - Quickly spot translation errors
✅ **Shareable** - Export TensorBoard logs to share results

---

## Tips:

1. **Increase num_examples** if you want to see more translations:
   ```python
   run_validation(..., num_examples=5)  # Show 5 examples instead of 2
   ```

2. **Long texts are truncated** to 50 characters in TensorBoard table for readability
   - Full text is shown in the console output

3. **Compare across epochs** - TensorBoard keeps history, so you can see how the same examples improve

---

## What's Been Changed:

1. **dataset.py** - Added `src_text` and `tgt_text` to dataset output
2. **train.py** - Enhanced `run_validation()` to:
   - Extract source and target text from batches
   - Generate predictions with greedy decode
   - Log formatted table to TensorBoard
   - Display all three texts in console

Enjoy your enhanced TensorBoard visualization! 🎉

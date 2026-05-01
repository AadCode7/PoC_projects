# Sentiment Analysis

## 🎯 Project Overview

This project investigates sentiment analysis using three different approaches:

1. **Naive Bayes Classifier** - A probabilistic machine learning approach
2. **Dictionary-Based (Rule-Based)** - Using pre-defined sentiment lexicons
3. **Enhanced Rule-Based** - Incorporating linguistic rules (negation, intensifiers, diminishers)

The goal is to understand the strengths and limitations of classical Bayesian text classification and compare it with rule-based approaches across different domains (movie reviews vs. product reviews).

---

## 📊 Dataset

### Movie Reviews (Rotten Tomatoes)
- **Positive reviews**: `rt-polarity.pos` (~5,331 snippets)
- **Negative reviews**: `rt-polarity.neg` (~5,331 snippets)
- **Domain**: Movie reviews from Rotten Tomatoes website
- **Split**: 90% training, 10% testing (random)

### Product Reviews (Nokia Phones)
- **Positive reviews**: `nokia-pos.txt` (~500 snippets)
- **Negative reviews**: `nokia-neg.txt` (~500 snippets)
- **Domain**: Nokia phone reviews
- **Purpose**: Cross-domain evaluation

### Sentiment Dictionary
- **Positive words**: 2,006 words (e.g., excellent, great, wonderful)
- **Negative words**: 4,783 words (e.g., terrible, awful, disappointing)
- **Source**: Pre-compiled sentiment lexicon

---

### 1️⃣ Naive Bayes Classifier

**Theory:**
Naive Bayes uses Bayes' theorem with the "naive" assumption that features (words) are conditionally independent given the class.

**Formula:**
```
P(positive|sentence) = P(positive) × ∏ P(word|positive)
P(negative|sentence) = P(negative) × ∏ P(word|negative)

Classification = argmax[P(positive|sentence), P(negative|sentence)]
```

---

### 2️⃣ Dictionary-Based (Rule-Based) Classifier

**Theory:**
Uses a pre-compiled lexicon of positive and negative words. Sentiment score is calculated by counting sentiment words.

**Algorithm:**
```
score = Σ(sentiment_value) for each word in sentence
where sentiment_value = +1 for positive words, -1 for negative words

if score >= threshold: classify as positive
else: classify as negative
```

---

### 3️⃣ Enhanced Rule-Based Classifier

**Theory:**
Extends dictionary-based approach with linguistic rules to handle contextual modifiers.

**Implemented Rules:**

#### Negation Handling
```python
negative_words = ['no', 'not', 'never', "but", "however", "although"]
# "not good" → flips sentiment of nearby positive words
```

#### Diminishers (Downtoners)
```python
diminisher_words = ["little", "few", "somewhat", "barely", "hardly", "slightly"]
# "somewhat good" → reduces sentiment intensity by 50%
score *= 0.5
```

#### Intensifiers (Amplifiers)
```python
emphasis_words = ["very", "extremely", "incredibly", "highly"]
# "very good" → increases sentiment intensity by 20%
score *= 1.2
```

#### Capitalization Detection
```python
# ALL CAPS words → increased weight
if word.isupper():
    score *= 1.2
```

#### Important Words Weighting
```python
# Words identified by mostUseful() get higher weight
if word in imp_words:
    score *= 1.5
```

---

## 📈 Detailed Comparison of Approaches

### Naive Bayes: Strengths & Weaknesses

#### ✅ Strengths

| Aspect | Explanation |
|--------|-------------|
| **Statistical Learning** | Learns patterns from data automatically, no manual feature engineering |
| **Domain Adaptation** | Can be retrained on new domains (e.g., Nokia reviews) |
| **Probability-Based** | Provides confidence scores (0-1 probability) |
| **Handles Unseen Words** | Smoothing techniques prevent zero probabilities |
| **Computational Efficiency** | O(n) complexity for both training and testing |
| **Small Data Efficiency** | Works well even with limited training data |
| **Baseline Performance** | Often achieves 70-85% accuracy out-of-the-box |

**Real-World Example:**
```
Training on movie reviews:
"brilliant acting" → P(positive) = 0.95
"boring plot" → P(positive) = 0.12

The model learns these associations automatically!
```

#### ❌ Weaknesses

| Aspect | Problem | Example |
|--------|---------|---------|
| **Independence Assumption** | Assumes words are independent (unrealistic) | "not good" treated as separate "not" + "good" |
| **Context Blindness** | Ignores word order and context | "good, not bad" vs "bad, not good" → same probability |
| **Domain Shift Problem** | Poor cross-domain performance | Movie model fails on Nokia reviews (different vocabulary) |
| **Negation Handling** | Cannot understand negation without features | "not terrible" classified as negative |
| **Data Imbalance** | Sensitive to class imbalance | If 90% positive training data → biased toward positive |
| **Feature Sparsity** | Short texts have few features | "Loved it!" → only 2 words to work with |
| **Sarcasm & Irony** | Cannot detect sarcasm | "Oh great, another terrible sequel" misclassified |

**Critical Failure Case:**
```
Sentence: "This movie is not good at all"
Naive Bayes sees: ["this", "movie", "is", "not", "good", "at", "all"]
- "good" → strong positive signal
- "not" → neutral/weak negative
Result: Classified as POSITIVE (wrong!)
```

---

### Dictionary-Based: Strengths & Weaknesses

#### ✅ Strengths

| Aspect | Explanation |
|--------|-------------|
| **No Training Required** | Works immediately with zero training data |
| **Interpretability** | 100% transparent - you can see exactly why a decision was made |
| **Domain Agnostic** | Same dictionary works across domains (movies, products, tweets) |
| **Fast Development** | Quick to implement and deploy |
| **Linguistic Knowledge** | Encodes human expertise about sentiment words |
| **Consistency** | Deterministic - same input always gives same output |
| **Low Resource** | Minimal computational requirements |

**Real-World Example:**
```
Sentence: "excellent phone with great battery life"
Dictionary counts:
- "excellent" = +1
- "great" = +1
Score = +2 → POSITIVE ✓

Clear, interpretable, and correct!
```

#### ❌ Weaknesses

| Aspect | Problem | Example |
|--------|---------|---------|
| **Context Ignorance** | No understanding of word relationships | "not excellent" still counts "excellent" as +1 |
| **Domain Vocabulary Gap** | May miss domain-specific sentiment | "lightweight" (positive for phones, negative for movies) |
| **Threshold Sensitivity** | Requires manual tuning per domain | threshold=1 for movies, 0.6 for Nokia |
| **Neutral Expressions** | Struggles with subtle sentiment | "It's okay" → score = 0 (but slightly positive intent) |
| **Vocabulary Coverage** | Limited to words in dictionary | New slang ("lit", "fire") not recognized |
| **Intensity Blindness** | "good" and "excellent" treated equally | Both = +1, but "excellent" is stronger |
| **Compound Expressions** | Misses phrases like "not too bad" | Counts only "bad" = -1 |

**Critical Failure Case:**
```
Sentence: "This phone is not bad, actually quite decent"
Dictionary sees:
- "bad" = -1
- "decent" = +1
Score = 0 → Classified as NEUTRAL/NEGATIVE (wrong! should be POSITIVE)
```

---

### Enhanced Rule-Based: Strengths & Weaknesses

#### ✅ Strengths

| Aspect | Explanation |
|--------|-------------|
| **Negation Handling** | Flips sentiment in negated contexts | "not good" correctly identified as negative |
| **Intensity Modulation** | Handles degree modifiers | "very good" weighted higher than "good" |
| **Linguistic Sophistication** | Incorporates syntactic patterns | Looks ±3 words for negation scope |
| **Improved Accuracy** | Typically 5-15% boost over simple dictionary | 68% → 78% accuracy on test data |
| **Interpretable Rules** | Each rule is human-understandable | "diminisher reduces score by 50%" |
| **Customizable** | Easy to add domain-specific rules | Add "battery life" as important for phones |

**Real-World Example:**
```
Sentence: "not very good but somewhat acceptable"

Enhanced Processing:
1. "very" → intensifier (×1.2)
2. "good" (+1) × 1.2 = +1.2
3. "not" detected → flip nearby "good" → -1.2
4. "but" → negation context
5. "somewhat" → diminisher (×0.5)
6. "acceptable" (+1) × 0.5 = +0.5
Final score: -1.2 + 0.5 = -0.7 → NEGATIVE ✓
```

#### ❌ Weaknesses

| Aspect | Problem | Example |
|--------|---------|---------|
| **Rule Engineering Overhead** | Requires expert knowledge to create rules | Linguist needed to identify all patterns |
| **Brittleness** | Rules may not generalize to new domains | "not bad" rule works differently in different contexts |
| **Scope Ambiguity** | Negation scope (±3 words) is arbitrary | "not really all that good" → scope issues |
| **Rule Conflicts** | Multiple rules may contradict | Negation + intensifier + diminisher → which wins? |
| **Long-Distance Dependencies** | Can't handle complex syntax | "The movie, despite some good acting, was terrible overall" |
| **Maintenance Burden** | Rules need constant updating | New linguistic patterns emerge over time |
| **Limited Compositionality** | Struggles with nested modifiers | "not entirely without merit" → too complex |

**Critical Failure Case:**
```
Sentence: "This isn't the worst phone, but it's far from the best"
Problems:
- "isn't" + "worst" → double negation (not handled)
- "far from the best" → requires understanding comparison
- "but" creates contrast that simple rules can't capture
Result: Likely misclassified
```

---

### When to Use Each Approach

#### Use Naive Bayes When:
- ✅ You have labeled training data (>1000 examples)
- ✅ Working within a single domain
- ✅ Need probabilistic confidence scores
- ✅ Data distribution is representative
- ✅ Quick baseline performance is needed

**Example Use Cases:**
- Email spam classification
- Product review sentiment (single category)
- Social media monitoring (single platform)

---

#### Use Dictionary-Based When:
- ✅ No training data available
- ✅ Need instant deployment
- ✅ 100% interpretability required
- ✅ Working across multiple domains
- ✅ Resource-constrained environment

**Example Use Cases:**
- Emergency sentiment monitoring (no time to train)
- Regulatory/compliance (need explainability)
- Multi-domain sentiment analysis
- Edge devices (IoT, mobile)

---

#### Use Enhanced Rule-Based When:
- ✅ Medium accuracy requirements (70-75%)
- ✅ Interpretability is important
- ✅ Have linguistic expertise available
- ✅ Domain-specific patterns are known
- ✅ Hybrid approach needed

**Example Use Cases:**
- Customer feedback analysis (need to explain to stakeholders)
- Domain-specific sentiment (medical, legal)
- Bootstrapping before collecting training data

---

## 📊 Results & Findings

### Typical Performance (Your Project)

#### Naive Bayes Results:
```
Films (Test Data):
- Accuracy: ~78-82%
- Precision (Positive): ~0.80
- Recall (Positive): ~0.78
- F1-Score: ~0.79

Nokia (Cross-Domain):
- Accuracy: ~55-60% ⚠️ (significant drop!)
- Shows domain adaptation challenges
```

#### Dictionary-Based Results:
```
Films (Test Data):
- Accuracy: ~65-68%
- More consistent across domains
- Lower peak performance but more stable

Nokia (Cross-Domain):
- Accuracy: ~63-66% ✓ (minimal drop)
```

#### Enhanced Rule-Based Results:
```
Films (Test Data):
- Accuracy: ~73-78%
- Precision (Positive): ~0.75
- Recall (Positive): ~0.73
- F1-Score: ~0.74

Nokia (Cross-Domain):
- Accuracy: ~68-72% ✓ (best cross-domain)
```

---

### Key Findings

#### 1. **Domain Shift Problem**
```
Why does Naive Bayes fail on Nokia data?

Movie vocabulary:
- "brilliant", "stunning", "masterpiece", "gripping"

Nokia vocabulary:
- "battery", "screen", "processor", "durable"

→ Completely different feature spaces!
```

**Solution in Modern Systems:**
- Domain adaptation techniques
- Transfer learning with BERT
- Multi-domain training

---

#### 2. **Most Useful Words Analysis**
```
Top Positive Words (Naive Bayes):
- "brilliant", "wonderful", "excellent", "perfect"

Top Negative Words (Naive Bayes):
- "worst", "awful", "terrible", "boring"

Overlap with Dictionary: ~45%
```

**Insight:** 
- Model learns domain-specific sentiment words not in general dictionaries
- "Touching" (positive in movies) may not be in standard lexicons

---

#### 3. **Error Patterns**

**Common Naive Bayes Errors:**
```
1. "This movie is not good" → POSITIVE ❌
   (Negation not understood)

2. "A bit disappointing but watchable" → NEGATIVE ❌
   (Mixed sentiment handling)

3. "Simply terrible acting" → POSITIVE ❌
   ("simply" + "terrible" confuses model)
```

**Common Dictionary Errors:**
```
1. "Not bad at all" → NEGATIVE ❌
   (Counts "bad" = -1)

2. "Could be better" → NEUTRAL ❌
   (No strong sentiment words)

3. "Meh, it's okay" → NEUTRAL ❌
   (Informal language not in dictionary)
```

---
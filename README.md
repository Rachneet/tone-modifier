## GPT Tone Modifier

A program that modifies the tone of text using a GPT model. 
It can be used to change the tone of sentences to be more simple or a direct tone.


### :rocket: Getting Started :rocket:

1. Create environment and install dependencies:
```bash
# create a virtual environment (e.g. conda)
conda create -n tone_gpt python=3.10
conda activate tone_gpt

# install the requirements
pip install -r requirements.txt
```


### Directory Structure

```
explaino/
├── 📂 data/
│   ├── 📄 fewshots.json                      # Few-shot examples for tone modification
│   ├── 📄 direct_transformed.json            # Direct tone modified data
│   ├── 📄 simple_transformed.json            # Simple tone modified data
│   ├── 📄 direct_transformed_2_shots.json                       
│   ├── 📄 simple_transformed_2_shots.json                       
│   ├── 📄 direct_transformed_5_shots.json                     
│   └── 📄 simple_transformed_5_shots.json                      
├── 📂 src/
│   ├── 🐍 model_client.py                    # Model client for interacting with the GPT model
│   ├── 🐍 preprocess.py                      # Data preprocessing scripts
│   ├── 🐍 evaluate.py                        # Evaluation scripts for model performance
│   ├── 🐍 config.py                          # Configuration settings for project
│   └── 🐍 prompts.py                         # Prompt templates
├── 🐍 main.py                                # Main application
├── 📄 .gitignore                             # Git ignore rules
├── 📄 README.md                              # Project documentation
└── 📄 requirements.txt                       # Python dependencies
```


### :memo: Usage :memo:

To run the application, use the following command:

1. If you want to use the exiting test data, use the following command:

```bash
## Arguments:
# --use_few_shots: Use few-shot examples for tone modification
# --tone: Specify the tone to modify to (e.g., 'simple', 'direct')
# --num_samples: Number few_shots to use (e.g., 2, 5)
# --few_shot_strategy : Strategy for few-shot examples (e.g., 'random', 'balanced')
# default setting is balanced to choose same no. of few-shot examples from each class when possible

## Example with few-shots
python main.py --use_few_shots --tone direct --num_samples 5

## Example without few-shots
python main.py --tone simple

## Results are saved in the data directory in the format of {tone}_transformed_{num_samples}_shots_{timestamp}.json
```

2. If you want to use your own data, you can use the following command:

```bash
## Arguments:
# --use_few_shots: Use few-shot examples for tone modification
# --tone: Specify the tone to modify to (e.g., 'simple', 'direct')
# --num_samples: Number few_shots to use (e.g., 2, 5)
python main.py --use_few_shots --tone direct --num_samples 5 --title "Your Title" --text "Your text to modify"
```

#### :memo: Example Input and Output
```bash
{
    "input": {
      "title": "Introduction to Environmental Sustainability at Work",
      "text": "Environmental sustainability in the workplace means reducing the negative impact of business activities on the environment. This includes conserving resources, minimizing waste, and lowering emissions. Companies that adopt sustainable practices not only help protect the planet but also often save costs, improve efficiency, and strengthen their public image."
    },
    "output": {
      "Title": "Introduction to Environmental Sustainability at Work",
      "Text": "Reduce your workplace’s environmental impact. Conserve resources, minimize waste, and lower emissions in all business activities. Adopt sustainable practices to protect the planet, cut costs, boost efficiency, and enhance your company’s public image. Take action now to drive positive change."
    }
  }
```

### Evaluation

1. **Structure Validation**: Validates that generated responses conform to the expected JSON structure.

Expected format:
```json
{
    "Title": "Modified Title",
    "Text": "Modified content text"
}
```

2. **Length Analysis**: Analyzes word count statistics for both input and output texts.

3. **Similarity Assessment**: Evaluates how similar generated responses are to expected outputs using both exact matching and semantic similarity.

#### Usage
To run the evaluation script, use the following command:

```bash 
## Arguments
# --results_path: Path to the results file to evaluate
# --metric: Metric to evaluate (e.g., 'length', 'similarity')

# Example for length evaluation
python src/evaluate.py --results_path ./data/simple_transformed.json --metric length

# Expected output format:
# {
# 'average_input_length': 41.2, 
# 'max_input_length': 48, 
# 'min_input_length': 34, 
# 'average_output_length': 39.2, 
# 'max_output_length': 44, 
# 'min_output_length': 34, 
# 'total_entries_analyzed': 5, 
# 'entries_with_errors': 0, 
# 'average_length_ratio': 0.9514563106796117
# }


# Example for similarity evaluation
# You need Spacy model installed for this
python -m spacy download en_core_web_lg
python src/evaluate.py --results_path ./data/simple_transformed.json --metric similarity

# Expected output format:
# {'title_matches': 5, 
# 'title_match_percentage': 100.0, 
# 'text_matches': 5, 
# 'text_match_percentage': 100.0, 
# 'average_similarity_score': 0.9241664290428162, 
# 'total_entries_analyzed': 5, 
# 'entries_with_errors': 0, 
# 'similarity_threshold_used': 0.8
# }
```

### Observations

General observations:

1. The model outputs the correct JSON structure with "Title" and "Text" keys.
2. The average word count for input and output texts is similar, indicating that the model maintains the original content length while modifying the tone.
3. The model achieves high similarity scores, indicating that the modified text retains the original meaning while changing the tone.

Insightful observations:

1. Model always generates text shorter than the input text. Interestingly, direct tone modification results in shorter outputs compared to simple tone modification.
2. The model is good at following instructions and never modifies the title as instructed.
3. Length of direct tone outputs is generally shorter than simple tone outputs, indicating that the model assumes a more concise and commanding style when modifying to a direct tone.
4. Similarity of direct tone outputs to original inputs is higher than simple tone outputs, indicating that the model is more consistent in maintaining the original meaning when modifying to a direct tone.
This is expected because the model simplifies the text more when changing to a simple tone, which may lead to more significant changes in wording and structure.

Effect of few-shot examples:
1. Generally, using few-shot examples improves the model's ability to modify tone effectively while maintaining the original meaning.


### Results

#### Direct tone modification results:
1. Similarity Evaluation Results

| Configuration | Entries | Title Matches | Text Matches | Avg Similarity | Threshold | Status |
|---------------|---------|---------------|--------------|----------------|-----------|--------|
| **Zero-shot** | 5 | 5 (100%) | 5 (100%) | 0.981 | 0.8 | ✅ Perfect |
| **2-shot** | 5 | 5 (100%) | 5 (100%) | 0.974 | 0.8 | ✅ Perfect |
| **5-shot** | 5 | 5 (100%) | 5 (100%) | 0.973 | 0.8 | ✅ Perfect |

> **Key Insight**: Results show excellent model performance across all few-shot configurations 
> with zero-shot surprisingly outperforming few-shot approaches.

2. Length Evaluation Results

| Configuration | Avg Input | Avg Output | Input Range | Output Range | Length Ratio | Entries | Status |
|---------------|-----------|------------|-------------|--------------|--------------|---------|--------|
| **Zero-shot** | 41.2 | 35.4 | 34-48 | 28-40 | 0.859 | 5 | ✅ |
| **2-shot** | 41.2 | 35.2 | 34-48 | 29-40 | 0.854 | 5 | ✅ |
| **5-shot** | 41.2 | 36.8 | 34-48 | 33-40 | **0.893** | 5 | ✅ |

> **Key Insight**: 5-shot configuration produces longer, more detailed outputs while maintaining quality, 
> with the highest length retention ratio (89.3%).


#### Simple tone modification results:
1. Similarity Evaluation Results

| Configuration | Entries | Title Matches | Text Matches | Avg Similarity | Threshold | Status |
|---------------|---------|---------------|--------------|----------------|-----------|--------|
| **Zero-shot** | 5 | 5 (100%) | 5 (100%) | **0.924** | 0.8 | ✅ Perfect |
| **2-shot** | 5 | 5 (100%) | 5 (100%) | 0.909 | 0.8 | ✅ Perfect |
| **5-shot** | 5 | 5 (100%) | 5 (100%) | 0.916 | 0.8 | ✅ Perfect |

> **Key Insight**: While simple tone maintains excellent performance (92%+ similarity), 
> direct tone achieves higher semantic similarity scores (97%+). 
> This suggests direct tone preserves meaning more precisely, while simple tone focuses on accessibility 
> with slight semantic trade-offs.

   
2. Length Evaluation Results

| Configuration | Avg Input | Avg Output | Input Range | Output Range | Length Ratio | Entries | Status |
|---------------|-----------|------------|-------------|--------------|--------------|---------|--------|
| **Zero-shot** | 41.2 | **39.2** | 34-48 | 34-44 | **0.951** | 5 | ✅ |
| **2-shot** | 41.2 | 36.6 | 34-48 | 32-41 | 0.888 | 5 | ✅ |
| **5-shot** | 41.2 | 38.2 | 34-48 | 34-43 | 0.927 | 5 | ✅ |

 > **Key Insight**: Simple tone transformation preserves significantly more content than direct tone, 
 > with zero-shot achieving 95% length retention - the highest across all configurations tested.


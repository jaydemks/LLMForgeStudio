using LLMForgeStudio.App.Core.Dataset;
using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.Core.Project;

public sealed class ForgeProject
{
    public string Name { get; set; } = "Untitled LLM Forge Project";
    public string DatasetPath { get; set; } = string.Empty;
    public string DatasetText { get; set; } = string.Empty;
    public string SelectedSection { get; set; } = "Dataset";
    public TextCleanerConfig Cleaner { get; set; } = new();
    public TokenizerConfig Tokenizer { get; set; } = new();
    public ModelConfig Model { get; set; } = new();
    public TrainingConfig Training { get; set; } = new();
    public SamplingConfig Sampling { get; set; } = new();
    public string PythonPath { get; set; } = "python";
    public string RunDirectory { get; set; } = string.Empty;
    public string CheckpointPath { get; set; } = string.Empty;
    public string GenerationPrompt { get; set; } = "The morning";
    public string Notes { get; set; } = string.Empty;
}

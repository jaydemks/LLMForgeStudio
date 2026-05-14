using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.Core.Backend;

public sealed class BackendJobSpec
{
    public string JobType { get; set; } = "train";
    public string DatasetPath { get; set; } = string.Empty;
    public string OutputDirectory { get; set; } = "runs/default";
    public TokenizerConfig Tokenizer { get; set; } = new();
    public ModelConfig Model { get; set; } = new();
    public TrainingConfig Training { get; set; } = new();
    public SamplingConfig Sampling { get; set; } = new();
    public string ClusterProfileName { get; set; } = "single-node";
    public string ClusterJobDescriptorPath { get; set; } = string.Empty;
}

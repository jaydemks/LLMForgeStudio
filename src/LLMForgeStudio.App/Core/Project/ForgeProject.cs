using LLMForgeStudio.App.Core.Dataset;
using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;
using System.Collections.Generic;

namespace LLMForgeStudio.App.Core.Project;

public sealed class ForgeProject
{
    public string Name { get; set; } = "Untitled LLM Forge Project";
    public string DatasetPath { get; set; } = string.Empty;
    public string DatasetText { get; set; } = string.Empty;
    public bool DatasetUsesExternalSource { get; set; }
    public string DatasetExternalSourcePath { get; set; } = string.Empty;
    public string SelectedSection { get; set; } = "Dataset";
    public TextCleanerConfig Cleaner { get; set; } = new();
    public TokenizerConfig Tokenizer { get; set; } = new();
    public ModelConfig Model { get; set; } = new();
    public TrainingConfig Training { get; set; } = new();
    public string SelectedTrainingProfile { get; set; } = "Custom";
    public SamplingConfig Sampling { get; set; } = new();
    public string PythonPath { get; set; } = "python";
    public string RunDirectory { get; set; } = string.Empty;
    public string CheckpointPath { get; set; } = string.Empty;
    public string GenerationPrompt { get; set; } = "The morning";
    public string Notes { get; set; } = string.Empty;
    public GatherProjectState Gather { get; set; } = new();
    public WorkflowProjectState Workflow { get; set; } = new();
}

public sealed class WorkflowProjectState
{
    public bool WizardSetupDone { get; set; }
    public bool WizardDatasetImported { get; set; }
    public bool WizardTokenizerTrained { get; set; }
    public bool WizardPreviewBuilt { get; set; }
    public bool WizardTrainingStarted { get; set; }
    public bool WizardCheckpointSet { get; set; }
    public string TokenizerStatusText { get; set; } = "Tokenizer not trained yet.";
    public string BatchPreviewStatusText { get; set; } = "x/y preview not built yet.";
}

public sealed class GatherProjectState
{
    public string SourceInput { get; set; } = string.Empty;
    public string WorkspaceDirectory { get; set; } = string.Empty;
    public string StagedDatasetPath { get; set; } = string.Empty;
    public string ValidationText { get; set; } = "-";
    public string RecommendedTokenizer { get; set; } = "-";
    public string RecommendedTrainingProfile { get; set; } = "-";
    public string MergeComplianceText { get; set; } = "Merge compliance: pending";
    public string DedupPolicy { get; set; } = "line";
    public List<string> StagedSources { get; set; } = new();
    public List<GatherSourceProjectState> SourceEntries { get; set; } = new();
}

public sealed class GatherSourceProjectState
{
    public string Path { get; set; } = string.Empty;
    public string Provider { get; set; } = string.Empty;
    public string LicenseLabel { get; set; } = "unknown";
    public bool IsLicensePermitted { get; set; }
    public bool IsEnabled { get; set; } = true;
    public int Weight { get; set; } = 1;
}

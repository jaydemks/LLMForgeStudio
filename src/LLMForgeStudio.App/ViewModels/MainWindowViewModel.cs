using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Input;
using System.Threading;
using Avalonia;
using Avalonia.Styling;
using Avalonia.Threading;
using LLMForgeStudio.App.Core.Backend;
using LLMForgeStudio.App.Core.Alignment;
using LLMForgeStudio.App.Core.Dataset;
using LLMForgeStudio.App.Core.Diagnostics;
using LLMForgeStudio.App.Core.Export;
using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Cluster;
using LLMForgeStudio.App.Core.Guidance;
using LLMForgeStudio.App.Core.Hardware;
using LLMForgeStudio.App.Core.Project;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel : ObservableObject
{
    private const int DatasetUiPreviewLimitChars = 120_000;
    private const int TokenizerUiLogLimitChars = 24_000;

    private enum BackendFlavor
    {
        NvidiaCuda,
        AmdDirectMl,
        Cpu
    }

    private string _datasetText = "The morning sun rises over a small local model.\nThis is only sample text. Replace it with your dataset.";
    private string _datasetPath = string.Empty;
    private bool _datasetUsesExternalSource;
    private string _datasetExternalSourcePath = string.Empty;
    private string _selectedSection = "Hardware";
    private TokenizerKind _selectedTokenizerKind = TokenizerKind.Character;
    private TokenizationResult? _lastTokenization;
    private string _log = "Ready. Load a dataset or use the sample text, then train a tokenizer.";
    private string _pythonPath = "python";
    private string _runDirectory = Path.Combine(Environment.CurrentDirectory, "runs", "default");
    private string _checkpointPath = string.Empty;
    private string _generationPrompt = "The morning";
    private string _generatedText = string.Empty;
    private string _generationStatusText = "Generation idle.";
    private bool _isGenerating;
    private bool _isTraining;
    private Process? _trainingProcess;
    private CancellationTokenSource? _trainingMonitorCts;
    private bool _isBackendBootstrapping;
    private bool _backendUserInitiatedStartup;
    private bool _showStartupOverlay = true;
    private bool _showBusyOverlay;
    private bool _busyOverlayCancelable;
    private double _busyOverlayOpacity;
    private string _busyOverlayText = "Processing...";
    private string _busyOverlaySubText = "Please wait.";
    private CancellationTokenSource? _busyOverlayCts;
    private int _backendSetupProgress;
    private string _backendSetupStage = "Idle";
    private bool _wizardEnabled = true;
    private bool _showSettingsOverlay;
    private bool _showAboutOverlay;
    private bool _showLogsOverlay;
    private bool _showNotificationsOverlay;
    private double _logsOverlayWidth = 900;
    private double _logsOverlayHeight = 560;
    private int _unreadNotificationsCount;
    private bool _isEnglish = true;
    private bool _isLightTheme;
    private double _cpuUsage;
    private double _gpuUsage;
    private double _ramUsage;
    private double _diskUsage;
    private readonly DispatcherTimer _perfTimer;
    private bool _isPerfSampling;
    private HardwareSummary? _lastHardwareSummary;
    private long _lastDiskIoBytes;
    private DateTimeOffset _lastDiskIoSampleAtUtc = DateTimeOffset.MinValue;
    private bool _wizardSetupDone;
    private bool _wizardDatasetImported;
    private bool _wizardTokenizerTrained;
    private bool _wizardPreviewBuilt;
    private bool _wizardTrainingStarted;
    private bool _wizardCheckpointSet;
    private string _tokenizerStatusText = "Tokenizer not trained yet.";
    private string _batchPreviewStatusText = "x/y preview not built yet.";
    private bool _isTokenizerBusy;
    private double _tokenizerProgressValue;
    private string _tokenizerProgressText = "Tokenizer idle.";
    private string _tokenizerLiveStatsText = "-";
    private CancellationTokenSource? _tokenizerCts;
    private CancellationTokenSource? _gatherCts;
    private readonly Dictionary<TokenizerKind, double> _tokenizerThroughputCharsPerSec = new();
    private DateTimeOffset _lastTokenizerRuntimeLogAtUtc = DateTimeOffset.MinValue;
    private string _trainingStatusText = "Training idle.";
    private string _rlhfDraftPrompt = string.Empty;
    private string _rlhfDraftChosen = string.Empty;
    private string _rlhfDraftRejected = string.Empty;
    private DateTimeOffset? _trainingStartedAtUtc;
    private DateTimeOffset _trainingLastProgressAtUtc = DateTimeOffset.MinValue;
    private bool _showAdvancedTraining;
    private string _selectedTrainingProfile = "Custom";
    private string _gatherSourceInput = string.Empty;
    private string _gatherWorkspaceDirectory = string.Empty;
    private string _gatherStatusText = "Gather Dataset workspace idle.";
    private bool _gatherIsBusy;
    private double _gatherProgressValue;
    private string _gatherProgressText = "Idle";
    private string _gatherValidationText = "-";
    private string _gatherLicenseText = "License check: pending";
    private bool _gatherLicensePermitted;
    private bool _gatherLicenseAcknowledged;
    private string _gatherHfDatasetId = string.Empty;
    private string _gatherRecommendedTokenizer = "-";
    private string _gatherRecommendedTrainingProfile = "-";
    private string _datasetRecommendedTokenizer = "-";
    private string _datasetRecommendedTrainingProfile = "-";
    private string _datasetRecommendationsText = "-";
    private string _gatherStagedDatasetPath = string.Empty;
    private string _gatherSourceProviderText = "Provider: -";
    private string _gatherMergeComplianceText = "Merge compliance: pending";
    private string _gatherDedupPolicy = "line";
    private string _clusterLiveSummaryText = "Cluster live status: idle.";
    private string _clusterLiveRoleText = "Role: standalone";
    private string _clusterLiveCoordinatorText = "Coordinator link: n/a";
    private string _ollamaFtBaseModelPath = string.Empty;
    private string _ollamaFtOutputModelName = "llmforge-italian-chat-v1";
    private string _ollamaFtDatasetPath = string.Empty;
    private string _ollamaFtOutputDirectory = string.Empty;
    private string _ollamaFtTemplate = "chatml";
    private string _ollamaFtMethod = "qlora";
    private string _ollamaFtBackend = "ollama-local";
    private int _ollamaFtEpochs = 3;
    private int _ollamaFtBatchSize = 2;
    private int _ollamaFtGradientAccumulation = 8;
    private double _ollamaFtLearningRate = 0.0002;
    private int _ollamaFtLoraRank = 16;
    private int _ollamaFtLoraAlpha = 32;
    private double _ollamaFtLoraDropout = 0.05;
    private bool _ollamaFtPackForOllama = true;
    private string _ollamaFtStatusText = "Fine-tuning workspace ready.";
    private bool _isOllamaFtRunning;
    private Process? _ollamaFtProcess;
    private EvalSummarySnapshot? _lastEvalSummary;
    private readonly UiDebugLogger _uiDebugLogger = new();

    public ObservableCollection<string> Sections { get; } = new(new[] { "Hardware", "Gather Dataset", "Dataset", "Tokenization", "Model", "Training", "Generation", "Fine-Tuning (Ollama, Experimental)" });
    public ObservableCollection<TokenizerKind> TokenizerKinds { get; } = new(Enum.GetValues<TokenizerKind>());
    public ObservableCollection<VocabularyItem> VocabularyPreview { get; } = new();
    public ObservableCollection<string> Warnings { get; } = new();
    public ObservableCollection<TrainingLogEntry> TrainingLogs { get; } = new();
    public ObservableCollection<string> HardwareGpus { get; } = new();
    public ObservableCollection<string> OptimizerOptions { get; } = new(new[] { "adamw", "lion", "adafactor" });
    public ObservableCollection<string> SchedulerOptions { get; } = new(new[] { "none", "cosine", "linear" });
    public ObservableCollection<string> PrecisionOptions { get; } = new(new[] { "fp16", "bf16" });
    public ObservableCollection<string> MultiGpuStrategyOptions { get; } = new(new[] { "none", "ddp", "fsdp" });
    public ObservableCollection<string> QuantizationProfileOptions { get; } = new(new[] { "dynamic-int8", "ptq-int8", "ptq-int4" });
    public ObservableCollection<string> AlignmentOptions { get; } = new(new[] { "none", "sft", "dpo", "rlhf" });
    public ObservableCollection<string> RlhfFeedbackOptions { get; } = new(new[] { "inline", "jsonl-human-feedback", "external-import" });
    public ObservableCollection<string> SafetyPolicyOptions { get; } = new(new[] { "standard", "strict", "research" });
    public ObservableCollection<string> EvalSuiteOptions { get; } = new(new[] { "basic", "quick-5", "standard-10", "full-20" });
    public ObservableCollection<string> TrainingProfileOptions { get; } = new(new[] { "Custom", "Tiny", "Balanced", "Serious", "Research", "Cluster" });
    public ObservableCollection<string> ClusterProfileOptions { get; } = new(ClusterProfileManager.BuiltIns.Select(x => x.Name));
    public ObservableCollection<string> OllamaFtTemplateOptions { get; } = new(new[] { "chatml", "alpaca", "openchat", "custom" });
    public ObservableCollection<string> OllamaFtMethodOptions { get; } = new(new[] { "qlora", "lora" });
    public ObservableCollection<string> OllamaFtBackendOptions { get; } = new(new[] { "ollama-local" });
    public ObservableCollection<string> GatherStagedSources { get; } = new();
    public ObservableCollection<GatherSourceEntryViewModel> GatherSourceEntries { get; } = new();
    public ObservableCollection<string> GatherDedupPolicyOptions { get; } = new(new[] { "none", "line", "paragraph" });
    public ObservableCollection<RlhfFeedbackRecord> RlhfCollectedFeedback { get; } = new();
    public ObservableCollection<string> ClusterLiveNodes { get; } = new();
    public ObservableCollection<string> ClusterLiveRemoteGpus { get; } = new();
    public ObservableCollection<NotificationEntryViewModel> Notifications { get; } = new();

    public TextCleanerConfig Cleaner { get; } = new();
    public TokenizerConfig TokenizerConfig { get; } = new();
    public ModelConfig ModelConfig { get; } = new();
    public TrainingConfig TrainingConfig { get; } = new();
    public SamplingConfig SamplingConfig { get; } = new();

    public ICommand TrainTokenizerCommand { get; }
    public ICommand BuildBatchPreviewCommand { get; }
    public ICommand DryRunTrainingCommand { get; }
    public ICommand GeneratePreviewCommand { get; }
    public ICommand StartBackendTrainingCommand { get; }
    public ICommand CancelBackendTrainingCommand { get; }
    public ICommand ExportForOllamaCommand { get; }
    public ICommand GenerateFromCheckpointCommand { get; }
    public ICommand SetupBackendCommand { get; }
    public ICommand DismissStartupOverlayCommand { get; }
    public ICommand ToggleSettingsCommand { get; }
    public ICommand ToggleLogsCommand { get; }
    public ICommand ToggleNotificationsCommand { get; }
    public ICommand ClearNotificationsCommand { get; }
    public ICommand ClearLogCommand { get; }
    public ICommand ToggleAboutCommand { get; }
    public ICommand CloseAboutCommand { get; }
    public ICommand ApplyLanguageItCommand { get; }
    public ICommand ApplyLanguageEnCommand { get; }
    public ICommand ToggleThemeCommand { get; }
    public ICommand SetLightThemeCommand { get; }
    public ICommand SetDarkThemeCommand { get; }
    public ICommand WizardGoCommand { get; }
    public ICommand DisableWizardCommand { get; }
    public ICommand AddRlhfFeedbackCommand { get; }
    public ICommand ClearRlhfFeedbackCommand { get; }
    public ICommand OpenGuideCommand { get; }
    public ICommand PrepareOllamaFineTuneCommand { get; }
    public ICommand StartOllamaFineTuneCommand { get; }
    public ICommand CancelOllamaFineTuneCommand { get; }
    public ICommand ConvertOllamaFineTuneToGgufCommand { get; }
    public ICommand FinalizeOllamaFineTuneExportCommand { get; }
    public ICommand GatherFetchSourceCommand { get; }
    public ICommand GatherCheckLicenseCommand { get; }
    public ICommand GatherConvertParquetCommand { get; }
    public ICommand GatherValidateDatasetCommand { get; }
    public ICommand GatherApplyRecommendationsCommand { get; }
    public ICommand GatherMergeSourcesCommand { get; }
    public ICommand GatherClearSourcesCommand { get; }
    public ICommand GatherHandoffToScratchCommand { get; }
    public ICommand GatherHandoffToFineTuneCommand { get; }
    public ICommand GatherCancelCommand { get; }
    public ICommand GatherRemoveSourceCommand { get; }
    public ICommand DatasetApplyRecommendationsCommand { get; }
    public ICommand CancelBusyOverlayCommand { get; }
    public ICommand CancelTokenizerCommand { get; }

    public MainWindowViewModel()
    {
        TrainTokenizerCommand = new RelayCommand(() => _ = TrainTokenizerAsync());
        BuildBatchPreviewCommand = new RelayCommand(() => _ = BuildBatchPreviewAsync());
        DryRunTrainingCommand = new RelayCommand(RunDryTraining);
        GeneratePreviewCommand = new RelayCommand(GeneratePreview);
        StartBackendTrainingCommand = new RelayCommand(() => _ = StartBackendTrainingAsync());
        CancelBackendTrainingCommand = new RelayCommand(CancelBackendTraining);
        ExportForOllamaCommand = new RelayCommand(() => _ = ExportForOllamaAsync());
        GenerateFromCheckpointCommand = new RelayCommand(() => _ = GenerateFromCheckpointAsync());
        SetupBackendCommand = new RelayCommand(() => _ = SetupBackendAsync());
        DismissStartupOverlayCommand = new RelayCommand(() => ShowStartupOverlay = false);
        ToggleSettingsCommand = new RelayCommand(() => ShowSettingsOverlay = !ShowSettingsOverlay);
        ToggleLogsCommand = new RelayCommand(() => ShowLogsOverlay = !ShowLogsOverlay);
        ToggleNotificationsCommand = new RelayCommand(ToggleNotificationsOverlay);
        ClearNotificationsCommand = new RelayCommand(ClearNotifications);
        ClearLogCommand = new RelayCommand(() => Log = string.Empty);
        ToggleAboutCommand = new RelayCommand(() => ShowAboutOverlay = true);
        CloseAboutCommand = new RelayCommand(() => ShowAboutOverlay = false);
        ApplyLanguageItCommand = new RelayCommand(() => SetLanguage(false));
        ApplyLanguageEnCommand = new RelayCommand(() => SetLanguage(true));
        ToggleThemeCommand = new RelayCommand(ToggleTheme);
        SetLightThemeCommand = new RelayCommand(() => ApplyTheme(true));
        SetDarkThemeCommand = new RelayCommand(() => ApplyTheme(false));
        WizardGoCommand = new RelayCommand(GoToWizardTarget);
        DisableWizardCommand = new RelayCommand(() => WizardEnabled = false);
        AddRlhfFeedbackCommand = new RelayCommand(AddRlhfFeedback);
        ClearRlhfFeedbackCommand = new RelayCommand(ClearRlhfFeedback);
        OpenGuideCommand = new RelayCommand(() => SelectedSection = IsEnglish ? "Guide" : "Guida");
        PrepareOllamaFineTuneCommand = new RelayCommand(PrepareOllamaFineTuneRun);
        StartOllamaFineTuneCommand = new RelayCommand(() => _ = StartOllamaFineTuneAsync());
        CancelOllamaFineTuneCommand = new RelayCommand(CancelOllamaFineTune);
        ConvertOllamaFineTuneToGgufCommand = new RelayCommand(() => _ = ConvertOllamaFineTuneToGgufAsync());
        FinalizeOllamaFineTuneExportCommand = new RelayCommand(FinalizeOllamaFineTuneExport);
        GatherFetchSourceCommand = new RelayCommand(() => _ = GatherFetchSourceAsync());
        GatherCheckLicenseCommand = new RelayCommand(() => _ = GatherCheckLicenseAsync());
        GatherConvertParquetCommand = new RelayCommand(() => _ = GatherConvertParquetAsync());
        GatherValidateDatasetCommand = new RelayCommand(() => _ = GatherValidateDatasetAsync());
        GatherApplyRecommendationsCommand = new RelayCommand(GatherApplyRecommendations);
        GatherMergeSourcesCommand = new RelayCommand(() => _ = GatherMergeSourcesAsync());
        GatherClearSourcesCommand = new RelayCommand(GatherClearSources);
        GatherHandoffToScratchCommand = new RelayCommand(() => _ = GatherHandoffToScratchAsync());
        GatherHandoffToFineTuneCommand = new RelayCommand(GatherHandoffToFineTune);
        GatherCancelCommand = new RelayCommand(CancelGatherOperation);
        GatherRemoveSourceCommand = new RelayCommand<GatherSourceEntryViewModel?>(RemoveGatherSource);
        DatasetApplyRecommendationsCommand = new RelayCommand(ApplyDatasetRecommendations);
        CancelBusyOverlayCommand = new RelayCommand(CancelBusyOverlay);
        CancelTokenizerCommand = new RelayCommand(CancelTokenizerRun);

        // Ensure initial tokenizer UI fields match the default selected tokenizer.
        ApplyRecommendedTokenizerSettings(_selectedTokenizerKind);

        RefreshAll();
        _ = RefreshHardwareAsync();
        _ = BootstrapBackendOnStartupAsync();

        _perfTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(3) };
        _perfTimer.Tick += async (_, _) =>
        {
            await RefreshPerformanceAsync();
            RefreshClusterLiveStatus();
        };
        _perfTimer.Start();
        UpdateSectionsForLanguage();
        RefreshClusterLiveStatus();
    }

    public string SelectedSection
    {
        get => _selectedSection;
        set
        {
            if (!SetProperty(ref _selectedSection, value)) return;
            OnPropertyChanged(nameof(IsDatasetSection));
            OnPropertyChanged(nameof(IsHardwareSection));
            OnPropertyChanged(nameof(IsGatherDatasetSection));
            OnPropertyChanged(nameof(IsTokenizationSection));
            OnPropertyChanged(nameof(IsModelSection));
            OnPropertyChanged(nameof(IsTrainingSection));
            OnPropertyChanged(nameof(IsGenerationSection));
            OnPropertyChanged(nameof(IsFineTuningOllamaSection));
            OnPropertyChanged(nameof(IsGuideSection));
        }
    }

    public string DatasetText
    {
        get => _datasetText;
        set
        {
            if (!SetProperty(ref _datasetText, value)) return;
            RefreshStats();
            OnPropertyChanged(nameof(TokenizerIdealValuesText));
        }
    }

    public string DatasetPath
    {
        get => _datasetPath;
        set => SetProperty(ref _datasetPath, value);
    }

    public string PythonPath
    {
        get => _pythonPath;
        set
        {
            if (!SetProperty(ref _pythonPath, value)) return;
            OnPropertyChanged(nameof(BackendStatusText));
            OnPropertyChanged(nameof(IsBackendConfigured));
            _ = RefreshHardwareAsync();
        }
    }

    public string RunDirectory
    {
        get => _runDirectory;
        set
        {
            if (!SetProperty(ref _runDirectory, value)) return;
            OnPropertyChanged(nameof(RunDirectoryHelpText));
            OnPropertyChanged(nameof(CanExportForOllama));
            OnPropertyChanged(nameof(OllamaExportButtonHint));
        }
    }

    public string CheckpointPath
    {
        get => _checkpointPath;
        set
        {
            if (!SetProperty(ref _checkpointPath, value)) return;
            _wizardCheckpointSet = !string.IsNullOrWhiteSpace(value);
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
        }
    }

    public string GenerationPrompt
    {
        get => _generationPrompt;
        set => SetProperty(ref _generationPrompt, value);
    }
    public string GeneratedText
    {
        get => _generatedText;
        private set => SetProperty(ref _generatedText, value);
    }
    public string GenerationStatusText
    {
        get => _generationStatusText;
        private set => SetProperty(ref _generationStatusText, value);
    }
    public bool IsGenerating
    {
        get => _isGenerating;
        private set
        {
            if (!SetProperty(ref _isGenerating, value)) return;
            OnPropertyChanged(nameof(IsGenerationIdle));
        }
    }
    public bool IsGenerationIdle => !IsGenerating;

    public bool IsTraining
    {
        get => _isTraining;
        set
        {
            if (!SetProperty(ref _isTraining, value)) return;
            OnPropertyChanged(nameof(IsTrainingIdle));
            OnPropertyChanged(nameof(TrainingStatusText));
            OnPropertyChanged(nameof(CanExportForOllama));
            OnPropertyChanged(nameof(OllamaExportButtonHint));
        }
    }

    public bool ShowStartupOverlay
    {
        get => _showStartupOverlay;
        set => SetProperty(ref _showStartupOverlay, value);
    }
    public bool ShowBusyOverlay
    {
        get => _showBusyOverlay;
        set => SetProperty(ref _showBusyOverlay, value);
    }
    public double BusyOverlayOpacity
    {
        get => _busyOverlayOpacity;
        set => SetProperty(ref _busyOverlayOpacity, Math.Clamp(value, 0, 1));
    }
    public string BusyOverlayText
    {
        get => _busyOverlayText;
        set => SetProperty(ref _busyOverlayText, value);
    }
    public string BusyOverlaySubText
    {
        get => _busyOverlaySubText;
        set => SetProperty(ref _busyOverlaySubText, value);
    }
    public bool BusyOverlayCancelable
    {
        get => _busyOverlayCancelable;
        private set => SetProperty(ref _busyOverlayCancelable, value);
    }
    public bool WizardEnabled
    {
        get => _wizardEnabled;
        set
        {
            if (!SetProperty(ref _wizardEnabled, value)) return;
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowSettingsOverlay
    {
        get => _showSettingsOverlay;
        set
        {
            if (!SetProperty(ref _showSettingsOverlay, value)) return;
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowAboutOverlay
    {
        get => _showAboutOverlay;
        set
        {
            if (!SetProperty(ref _showAboutOverlay, value)) return;
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowLogsOverlay
    {
        get => _showLogsOverlay;
        set
        {
            if (!SetProperty(ref _showLogsOverlay, value)) return;
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowNotificationsOverlay
    {
        get => _showNotificationsOverlay;
        set
        {
            if (!SetProperty(ref _showNotificationsOverlay, value)) return;
            if (value)
            {
                UnreadNotificationsCount = 0;
                foreach (var n in Notifications)
                    n.IsUnread = false;
            }
            OnPropertyChanged(nameof(ShowWizardOverlay));
            OnPropertyChanged(nameof(NotificationsButtonText));
        }
    }
    public double LogsOverlayWidth
    {
        get => _logsOverlayWidth;
        set => SetProperty(ref _logsOverlayWidth, Math.Clamp(value, 640, 1600));
    }
    public double LogsOverlayHeight
    {
        get => _logsOverlayHeight;
        set => SetProperty(ref _logsOverlayHeight, Math.Clamp(value, 380, 1000));
    }
    public int UnreadNotificationsCount
    {
        get => _unreadNotificationsCount;
        private set
        {
            if (!SetProperty(ref _unreadNotificationsCount, Math.Max(0, value))) return;
            OnPropertyChanged(nameof(NotificationsButtonText));
        }
    }
    public bool IsEnglish
    {
        get => _isEnglish;
        set
        {
            if (!SetProperty(ref _isEnglish, value)) return;
            RefreshLocalizedTexts();
        }
    }
    public bool IsLightTheme
    {
        get => _isLightTheme;
        set => SetProperty(ref _isLightTheme, value);
    }

    public int BackendSetupProgress
    {
        get => _backendSetupProgress;
        set => SetProperty(ref _backendSetupProgress, value);
    }

    public string BackendSetupStage
    {
        get => _backendSetupStage;
        set => SetProperty(ref _backendSetupStage, value);
    }

    public TokenizerKind SelectedTokenizerKind
    {
        get => _selectedTokenizerKind;
        set
        {
            if (SetProperty(ref _selectedTokenizerKind, value))
            {
                TokenizerConfig.Kind = value;
                OnPropertyChanged(nameof(TokenizerExplanation));
                OnPropertyChanged(nameof(ShowBpeOptions));
                OnPropertyChanged(nameof(ShowWordOptions));
                OnPropertyChanged(nameof(ShowExperimentalOptions));
                ApplyRecommendedTokenizerSettings(value);
                ApplyRecommendedModelAndTrainingSettings(value);
                OnPropertyChanged(nameof(TokenizerConfig));
                OnPropertyChanged(nameof(TokenizerTargetVocabSize));
                OnPropertyChanged(nameof(TokenizerMinFrequency));
                OnPropertyChanged(nameof(TokenizerMaxMerges));
                OnPropertyChanged(nameof(TokenizerRecommendationText));
                OnPropertyChanged(nameof(TokenizerPresetBadgeText));
                OnPropertyChanged(nameof(TokenizerIdealValuesText));
                RaiseModelTrainingBindingsChanged();
                OnPropertyChanged(nameof(ParameterEstimate));
                RefreshWarnings();
            }
        }
    }

    public string Log
    {
        get => _log;
        set => SetProperty(ref _log, value);
    }

    public string TokenizerStatusText
    {
        get => _tokenizerStatusText;
        private set => SetProperty(ref _tokenizerStatusText, value);
    }

    public string BatchPreviewStatusText
    {
        get => _batchPreviewStatusText;
        private set => SetProperty(ref _batchPreviewStatusText, value);
    }

    public bool IsTokenizerBusy
    {
        get => _isTokenizerBusy;
        private set
        {
            if (!SetProperty(ref _isTokenizerBusy, value)) return;
            OnPropertyChanged(nameof(IsTokenizerIdle));
            OnPropertyChanged(nameof(CanBuildBatchPreview));
            OnPropertyChanged(nameof(CanCancelTokenizer));
        }
    }

    public bool IsTokenizerIdle => !IsTokenizerBusy;

    public double TokenizerProgressValue
    {
        get => _tokenizerProgressValue;
        private set => SetProperty(ref _tokenizerProgressValue, Math.Clamp(value, 0, 100));
    }

    public string TokenizerProgressText
    {
        get => _tokenizerProgressText;
        private set => SetProperty(ref _tokenizerProgressText, value);
    }
    public string TokenizerLiveStatsText
    {
        get => _tokenizerLiveStatsText;
        private set => SetProperty(ref _tokenizerLiveStatsText, value);
    }

    public bool CanBuildBatchPreview => IsTokenizerIdle;
    public bool CanCancelTokenizer => IsTokenizerBusy;

    public bool IsTokenizerReady => _wizardTokenizerTrained || _lastTokenization is not null;
    public bool IsBatchPreviewReady => _wizardPreviewBuilt;
    public bool IsTrainingReady => _wizardDatasetImported && IsTokenizerReady;
    public string TrainingReadinessText
        => IsTrainingReady
            ? "Ready for training."
            : $"Missing steps: {string.Join(", ", GetTrainingMissingSteps())}.";

    public string DatasetStatsText { get; private set; } = string.Empty;
    public string DatasetCharsText { get; private set; } = "0";
    public string DatasetLinesText { get; private set; } = "0";
    public string DatasetWordsText { get; private set; } = "0";
    public string DatasetUniqueCharsText { get; private set; } = "0";
    public int TokenizerTargetVocabSize
    {
        get => TokenizerConfig.TargetVocabSize;
        set
        {
            if (TokenizerConfig.TargetVocabSize == value) return;
            TokenizerConfig.TargetVocabSize = value;
            OnPropertyChanged();
        }
    }
    public int TokenizerMinFrequency
    {
        get => TokenizerConfig.MinFrequency;
        set
        {
            if (TokenizerConfig.MinFrequency == value) return;
            TokenizerConfig.MinFrequency = value;
            OnPropertyChanged();
        }
    }
    public int TokenizerMaxMerges
    {
        get => TokenizerConfig.MaxMerges;
        set
        {
            if (TokenizerConfig.MaxMerges == value) return;
            TokenizerConfig.MaxMerges = value;
            OnPropertyChanged();
        }
    }
    public int ModelVocabSize
    {
        get => ModelConfig.VocabSize;
        set
        {
            if (ModelConfig.VocabSize == value) return;
            ModelConfig.VocabSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelBlockSize
    {
        get => ModelConfig.BlockSize;
        set
        {
            if (ModelConfig.BlockSize == value) return;
            ModelConfig.BlockSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            OnPropertyChanged(nameof(ModelContextWindowText));
            RefreshWarnings();
        }
    }
    public int ModelLayers
    {
        get => ModelConfig.Layers;
        set
        {
            if (ModelConfig.Layers == value) return;
            ModelConfig.Layers = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelHeads
    {
        get => ModelConfig.Heads;
        set
        {
            if (ModelConfig.Heads == value) return;
            ModelConfig.Heads = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelEmbeddingSize
    {
        get => ModelConfig.EmbeddingSize;
        set
        {
            if (ModelConfig.EmbeddingSize == value) return;
            ModelConfig.EmbeddingSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int TrainingBatchSize
    {
        get => TrainingConfig.BatchSize;
        set
        {
            if (TrainingConfig.BatchSize == value) return;
            TrainingConfig.BatchSize = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public int TrainingMaxSteps
    {
        get => TrainingConfig.MaxSteps;
        set
        {
            if (TrainingConfig.MaxSteps == value) return;
            TrainingConfig.MaxSteps = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public double TrainingLearningRate
    {
        get => TrainingConfig.LearningRate;
        set
        {
            if (Math.Abs(TrainingConfig.LearningRate - value) < 1e-12) return;
            TrainingConfig.LearningRate = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public int TrainingEvalEvery
    {
        get => TrainingConfig.EvalEvery;
        set
        {
            if (TrainingConfig.EvalEvery == value) return;
            TrainingConfig.EvalEvery = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public bool ShowAdvancedTraining
    {
        get => _showAdvancedTraining;
        set => SetProperty(ref _showAdvancedTraining, value);
    }
    public string SelectedTrainingProfile
    {
        get => _selectedTrainingProfile;
        set
        {
            if (!SetProperty(ref _selectedTrainingProfile, value)) return;
            ApplyTrainingProfile(value);
            RaiseModelTrainingBindingsChanged();
            RefreshWarnings();
            OnPropertyChanged(nameof(TrainingProfileHintText));
        }
    }
    public string TrainingProfileHintText => GuidedDefaultsEngine.DescribeTrainingProfile(SelectedTrainingProfile);
    public string TrainingOptimizer
    {
        get => TrainingConfig.Optimizer;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "adamw" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Optimizer == next) return;
            TrainingConfig.Optimizer = next;
            OnPropertyChanged();
        }
    }
    public string TrainingScheduler
    {
        get => TrainingConfig.Scheduler;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Scheduler == next) return;
            TrainingConfig.Scheduler = next;
            OnPropertyChanged();
        }
    }
    public int TrainingWarmupSteps
    {
        get => TrainingConfig.WarmupSteps;
        set
        {
            if (TrainingConfig.WarmupSteps == value) return;
            TrainingConfig.WarmupSteps = Math.Max(0, value);
            OnPropertyChanged();
        }
    }
    public bool TrainingMixedPrecision
    {
        get => TrainingConfig.MixedPrecision;
        set
        {
            if (TrainingConfig.MixedPrecision == value) return;
            TrainingConfig.MixedPrecision = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingPrecision
    {
        get => TrainingConfig.Precision;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "fp16" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Precision == next) return;
            TrainingConfig.Precision = next;
            OnPropertyChanged();
        }
    }
    public bool TrainingGradientClipping
    {
        get => TrainingConfig.EnableGradientClipping;
        set
        {
            if (TrainingConfig.EnableGradientClipping == value) return;
            TrainingConfig.EnableGradientClipping = value;
            OnPropertyChanged();
        }
    }
    public double TrainingGradientClipNorm
    {
        get => TrainingConfig.GradientClipNorm;
        set
        {
            if (Math.Abs(TrainingConfig.GradientClipNorm - value) < 1e-12) return;
            TrainingConfig.GradientClipNorm = Math.Max(0.0, value);
            OnPropertyChanged();
        }
    }
    public int TrainingCheckpointEvery
    {
        get => TrainingConfig.CheckpointEvery;
        set
        {
            if (TrainingConfig.CheckpointEvery == value) return;
            TrainingConfig.CheckpointEvery = Math.Max(0, value);
            OnPropertyChanged();
        }
    }
    public bool TrainingEnablePostQuantization
    {
        get => TrainingConfig.EnablePostTrainingQuantization;
        set
        {
            if (TrainingConfig.EnablePostTrainingQuantization == value) return;
            TrainingConfig.EnablePostTrainingQuantization = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingQuantizationProfile
    {
        get => TrainingConfig.QuantizationProfile;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "dynamic-int8" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.QuantizationProfile == next) return;
            TrainingConfig.QuantizationProfile = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingQuantizationCalibrationSamples
    {
        get => TrainingConfig.QuantizationCalibrationSamples;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.QuantizationCalibrationSamples == next) return;
            TrainingConfig.QuantizationCalibrationSamples = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingEnableQatPath
    {
        get => TrainingConfig.EnableQatPath;
        set
        {
            if (TrainingConfig.EnableQatPath == value) return;
            TrainingConfig.EnableQatPath = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingQatFineTuneSteps
    {
        get => TrainingConfig.QatFineTuneSteps;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.QatFineTuneSteps == next) return;
            TrainingConfig.QatFineTuneSteps = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingDedup
    {
        get => TrainingConfig.EnableDeduplication;
        set
        {
            if (TrainingConfig.EnableDeduplication == value) return;
            TrainingConfig.EnableDeduplication = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingDedupLines
    {
        get => TrainingConfig.RemoveDuplicateLines;
        set
        {
            if (TrainingConfig.RemoveDuplicateLines == value) return;
            TrainingConfig.RemoveDuplicateLines = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingDedupParagraphs
    {
        get => TrainingConfig.RemoveDuplicateParagraphs;
        set
        {
            if (TrainingConfig.RemoveDuplicateParagraphs == value) return;
            TrainingConfig.RemoveDuplicateParagraphs = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingNormalizeUnicode
    {
        get => TrainingConfig.NormalizeUnicode;
        set
        {
            if (TrainingConfig.NormalizeUnicode == value) return;
            TrainingConfig.NormalizeUnicode = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingCollapseWhitespace
    {
        get => TrainingConfig.CollapseWhitespace;
        set
        {
            if (TrainingConfig.CollapseWhitespace == value) return;
            TrainingConfig.CollapseWhitespace = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingCurriculumLearning
    {
        get => TrainingConfig.CurriculumLearning;
        set
        {
            if (TrainingConfig.CurriculumLearning == value) return;
            TrainingConfig.CurriculumLearning = value;
            OnPropertyChanged();
        }
    }
    public double TrainingCurriculumWarmupRatio
    {
        get => TrainingConfig.CurriculumWarmupRatio;
        set
        {
            if (Math.Abs(TrainingConfig.CurriculumWarmupRatio - value) < 1e-12) return;
            TrainingConfig.CurriculumWarmupRatio = Math.Clamp(value, 0.05, 0.9);
            OnPropertyChanged();
        }
    }
    public bool TrainingDistributed
    {
        get => TrainingConfig.DistributedTraining;
        set
        {
            if (TrainingConfig.DistributedTraining == value) return;
            TrainingConfig.DistributedTraining = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingClusterProfileName
    {
        get => TrainingConfig.ClusterProfileName;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "single-node" : value.Trim();
            if (TrainingConfig.ClusterProfileName == next) return;
            TrainingConfig.ClusterProfileName = next;
            var profile = ClusterProfileManager.Resolve(next);
            TrainingConfig.ClusterOrchestrator = profile.Orchestrator;
            TrainingConfig.ClusterWorldSize = Math.Max(1, profile.WorldSize);
            TrainingConfig.ClusterMaxRetries = Math.Max(0, profile.MaxRetries);
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
            OnPropertyChanged(nameof(TrainingWizardDetailedText));
        }
    }
    public bool TrainingOrchestratePipelineStages
    {
        get => TrainingConfig.OrchestratePipelineStages;
        set
        {
            if (TrainingConfig.OrchestratePipelineStages == value) return;
            TrainingConfig.OrchestratePipelineStages = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunDataStage
    {
        get => TrainingConfig.PipelineRunDataStage;
        set
        {
            if (TrainingConfig.PipelineRunDataStage == value) return;
            TrainingConfig.PipelineRunDataStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunPreprocessStage
    {
        get => TrainingConfig.PipelineRunPreprocessStage;
        set
        {
            if (TrainingConfig.PipelineRunPreprocessStage == value) return;
            TrainingConfig.PipelineRunPreprocessStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunTrainStage
    {
        get => TrainingConfig.PipelineRunTrainStage;
        set
        {
            if (TrainingConfig.PipelineRunTrainStage == value) return;
            TrainingConfig.PipelineRunTrainStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunEvalStage
    {
        get => TrainingConfig.PipelineRunEvalStage;
        set
        {
            if (TrainingConfig.PipelineRunEvalStage == value) return;
            TrainingConfig.PipelineRunEvalStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingMultiGpuStrategy
    {
        get => TrainingConfig.MultiGpuStrategy;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.MultiGpuStrategy == next) return;
            TrainingConfig.MultiGpuStrategy = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingGradientAccumulationSteps
    {
        get => TrainingConfig.GradientAccumulationSteps;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.GradientAccumulationSteps == next) return;
            TrainingConfig.GradientAccumulationSteps = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingAutoDeviceMap
    {
        get => TrainingConfig.AutoDeviceMap;
        set
        {
            if (TrainingConfig.AutoDeviceMap == value) return;
            TrainingConfig.AutoDeviceMap = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingAlignmentMode
    {
        get => TrainingConfig.AlignmentMode;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.AlignmentMode == next) return;
            TrainingConfig.AlignmentMode = next;
            OnPropertyChanged();
        }
    }
    public bool TrainingFineTuningOrchestration
    {
        get => TrainingConfig.FineTuningOrchestration;
        set
        {
            if (TrainingConfig.FineTuningOrchestration == value) return;
            TrainingConfig.FineTuningOrchestration = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageSft
    {
        get => TrainingConfig.FineTuneStageSft;
        set
        {
            if (TrainingConfig.FineTuneStageSft == value) return;
            TrainingConfig.FineTuneStageSft = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageDpo
    {
        get => TrainingConfig.FineTuneStageDpo;
        set
        {
            if (TrainingConfig.FineTuneStageDpo == value) return;
            TrainingConfig.FineTuneStageDpo = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageRlhf
    {
        get => TrainingConfig.FineTuneStageRlhf;
        set
        {
            if (TrainingConfig.FineTuneStageRlhf == value) return;
            TrainingConfig.FineTuneStageRlhf = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingRlhfFeedbackSource
    {
        get => TrainingConfig.RlhfFeedbackSource;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "inline" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.RlhfFeedbackSource == next) return;
            TrainingConfig.RlhfFeedbackSource = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingRlhfFeedbackPath
    {
        get => TrainingConfig.RlhfFeedbackPath;
        set
        {
            if (TrainingConfig.RlhfFeedbackPath == value) return;
            TrainingConfig.RlhfFeedbackPath = value ?? string.Empty;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string RlhfDraftPrompt
    {
        get => _rlhfDraftPrompt;
        set => SetProperty(ref _rlhfDraftPrompt, value);
    }
    public string RlhfDraftChosen
    {
        get => _rlhfDraftChosen;
        set => SetProperty(ref _rlhfDraftChosen, value);
    }
    public string RlhfDraftRejected
    {
        get => _rlhfDraftRejected;
        set => SetProperty(ref _rlhfDraftRejected, value);
    }
    public string RlhfCollectedCountText => $"Collected RLHF records: {RlhfCollectedFeedback.Count}";
    public bool TrainingRewardModelingEnabled
    {
        get => TrainingConfig.RewardModelingEnabled;
        set
        {
            if (TrainingConfig.RewardModelingEnabled == value) return;
            TrainingConfig.RewardModelingEnabled = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingSafetyPolicyMode
    {
        get => TrainingConfig.SafetyPolicyMode;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "standard" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.SafetyPolicyMode == next) return;
            TrainingConfig.SafetyPolicyMode = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingExportOnnx
    {
        get => TrainingConfig.ExportOnnx;
        set
        {
            if (TrainingConfig.ExportOnnx == value) return;
            TrainingConfig.ExportOnnx = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingExportGguf
    {
        get => TrainingConfig.ExportGguf;
        set
        {
            if (TrainingConfig.ExportGguf == value) return;
            TrainingConfig.ExportGguf = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
            OnPropertyChanged(nameof(CanExportForOllama));
            OnPropertyChanged(nameof(OllamaExportButtonHint));
        }
    }
    public bool CanExportForOllama
    {
        get
        {
            if (IsTraining) return false;
            if (!TrainingConfig.ExportGguf) return false;
            if (string.IsNullOrWhiteSpace(RunDirectory)) return false;
            return OllamaExportPackager.IsReadyForPackaging(RunDirectory);
        }
    }
    public string OllamaExportButtonHint => CanExportForOllama
        ? "Export bundle pronto: crea/sovrascrive exports/ollama nel run corrente."
        : "Disponibile solo quando training completato con GGUF reale e opzione Export GGUF attiva.";
    public string TrainingEvalSuite
    {
        get => TrainingConfig.EvalSuite;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "basic" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.EvalSuite == next) return;
            TrainingConfig.EvalSuite = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TokenizerExplanation => TokenizerRegistry.Explain(SelectedTokenizerKind);
    public bool ShowBpeOptions => SelectedTokenizerKind is TokenizerKind.SimpleBpe or TokenizerKind.ByteLevelBpe or TokenizerKind.Unigram or TokenizerKind.WordPiece or TokenizerKind.HybridFallback;
    public bool ShowWordOptions => SelectedTokenizerKind is TokenizerKind.Word;
    public bool ShowExperimentalOptions => SelectedTokenizerKind is TokenizerKind.HierarchicalExperimental;
    public bool IsDatasetSection => SelectedSection == "Dataset";
    public bool IsGatherDatasetSection => SelectedSection is "Gather Dataset" or "Raccolta Dataset";
    public bool IsHardwareSection => SelectedSection == "Hardware";
    public bool IsTokenizationSection => SelectedSection == "Tokenization";
    public bool IsModelSection => SelectedSection == "Model";
    public bool IsTrainingSection => SelectedSection == "Training";
    public bool IsGenerationSection => SelectedSection is "Generation" or "Generazione";
    public bool IsFineTuningOllamaSection => SelectedSection is "Fine-Tuning (Ollama, Experimental)" or "Fine-Tuning (Ollama, Experimental)";
    public bool IsGuideSection => SelectedSection is "Guide" or "Guida";
    public string GatherSourceInput
    {
        get => _gatherSourceInput;
        set
        {
            if (!SetProperty(ref _gatherSourceInput, value)) return;
            var provider = ResolveGatherProviderContract(value ?? string.Empty);
            GatherSourceProviderText = $"Provider: {provider.DisplayName}";
            // Reset license gate state whenever source changes to avoid reusing
            // previous source approval on a new provider URL/path.
            _gatherHfDatasetId = string.Empty;
            GatherLicensePermitted = false;
            GatherLicenseAcknowledged = false;
            GatherLicenseText = IsEnglish ? "License check: pending" : "Controllo licenza: in attesa";
            OnPropertyChanged(nameof(CanGatherFetchSource));
        }
    }
    public string GatherWorkspaceDirectory
    {
        get => _gatherWorkspaceDirectory;
        set => SetProperty(ref _gatherWorkspaceDirectory, value);
    }
    public string GatherStatusText
    {
        get => _gatherStatusText;
        set => SetProperty(ref _gatherStatusText, value);
    }
    public bool GatherIsBusy
    {
        get => _gatherIsBusy;
        set
        {
            if (!SetProperty(ref _gatherIsBusy, value)) return;
            OnPropertyChanged(nameof(CanGatherFetchSource));
            OnPropertyChanged(nameof(CanGatherConvertParquet));
            OnPropertyChanged(nameof(CanGatherMergeSources));
            OnPropertyChanged(nameof(CanGatherValidateDataset));
            OnPropertyChanged(nameof(CanGatherApplyRecommendations));
            OnPropertyChanged(nameof(CanGatherHandoffScratch));
            OnPropertyChanged(nameof(CanGatherHandoffFineTune));
            OnPropertyChanged(nameof(CanCancelGather));
        }
    }
    public double GatherProgressValue
    {
        get => _gatherProgressValue;
        set => SetProperty(ref _gatherProgressValue, value);
    }
    public string GatherProgressText
    {
        get => _gatherProgressText;
        set => SetProperty(ref _gatherProgressText, value);
    }
    public string GatherValidationText
    {
        get => _gatherValidationText;
        set => SetProperty(ref _gatherValidationText, value);
    }
    public string GatherLicenseText
    {
        get => _gatherLicenseText;
        set => SetProperty(ref _gatherLicenseText, value);
    }
    public bool GatherLicensePermitted
    {
        get => _gatherLicensePermitted;
        set
        {
            if (!SetProperty(ref _gatherLicensePermitted, value)) return;
            OnPropertyChanged(nameof(CanGatherFetchSource));
        }
    }
    public bool GatherLicenseAcknowledged
    {
        get => _gatherLicenseAcknowledged;
        set
        {
            if (!SetProperty(ref _gatherLicenseAcknowledged, value)) return;
            OnPropertyChanged(nameof(CanGatherFetchSource));
        }
    }
    public bool CanGatherFetchSource
    {
        get
        {
            var src = (GatherSourceInput ?? string.Empty).Trim();
            if (string.IsNullOrWhiteSpace(src)) return false;
            if (GatherIsBusy) return false;
            var provider = ResolveGatherProviderContract(src);
            if (provider.Kind == GatherProviderKind.Unsupported) return false;
            if (provider.RequiresExplicitLicenseCheck)
                return GatherLicensePermitted && GatherLicenseAcknowledged;
            return GatherLicenseAcknowledged;
        }
    }
    public bool CanGatherConvertParquet => !GatherIsBusy && !string.IsNullOrWhiteSpace(GatherStagedDatasetPath) && GatherNeedsParquetConversion;
    public bool CanGatherMergeSources => !GatherIsBusy && GatherSourceEntries.Count(x => x.IsEnabled) > 0 && !GatherNeedsParquetConversion;
    public bool CanGatherValidateDataset => !GatherIsBusy && !string.IsNullOrWhiteSpace(GatherStagedDatasetPath) && !GatherNeedsParquetConversion;
    public bool CanGatherApplyRecommendations => !GatherIsBusy;
    public bool CanGatherHandoffScratch => !GatherIsBusy && !string.IsNullOrWhiteSpace(GatherStagedDatasetPath);
    public bool CanGatherHandoffFineTune => !GatherIsBusy && !string.IsNullOrWhiteSpace(GatherStagedDatasetPath);
    public bool CanCancelGather => GatherIsBusy;
    public bool GatherNeedsParquetConversion => DetectGatherParquetInputs();
    public string GatherParquetHintText => GatherNeedsParquetConversion
        ? (IsEnglish
            ? "Parquet detected in staged sources: run Step 3 (Convert Parquet) before Merge/Validate."
            : "Parquet rilevato nelle sorgenti: esegui Step 3 (Converti Parquet) prima di Merge/Validate.")
        : (IsEnglish
            ? "No parquet detected in staged sources: you can proceed to Merge."
            : "Nessun parquet rilevato: puoi procedere con Merge.");
    public string GatherRecommendedTokenizer
    {
        get => _gatherRecommendedTokenizer;
        set
        {
            if (!SetProperty(ref _gatherRecommendedTokenizer, value)) return;
            OnPropertyChanged(nameof(GatherRecommendedTokenizerLine));
        }
    }
    public string GatherRecommendedTrainingProfile
    {
        get => _gatherRecommendedTrainingProfile;
        set
        {
            if (!SetProperty(ref _gatherRecommendedTrainingProfile, value)) return;
            OnPropertyChanged(nameof(GatherRecommendedTrainingProfileLine));
        }
    }
    public string GatherStagedDatasetPath
    {
        get => _gatherStagedDatasetPath;
        set
        {
            if (!SetProperty(ref _gatherStagedDatasetPath, value)) return;
            OnPropertyChanged(nameof(GatherNeedsParquetConversion));
            OnPropertyChanged(nameof(GatherParquetHintText));
            OnPropertyChanged(nameof(CanGatherConvertParquet));
            OnPropertyChanged(nameof(CanGatherMergeSources));
            OnPropertyChanged(nameof(CanGatherValidateDataset));
            OnPropertyChanged(nameof(CanGatherHandoffScratch));
            OnPropertyChanged(nameof(CanGatherHandoffFineTune));
        }
    }
    public string GatherSourceProviderText
    {
        get => _gatherSourceProviderText;
        set => SetProperty(ref _gatherSourceProviderText, value);
    }
    public string GatherMergeComplianceText
    {
        get => _gatherMergeComplianceText;
        set => SetProperty(ref _gatherMergeComplianceText, value);
    }
    public string GatherDedupPolicy
    {
        get => _gatherDedupPolicy;
        set => SetProperty(ref _gatherDedupPolicy, string.IsNullOrWhiteSpace(value) ? "line" : value.Trim().ToLowerInvariant());
    }
    public string GatherSourcesCountText => IsEnglish
        ? $"Staged sources: {GatherStagedSources.Count}"
        : $"Sorgenti in staging: {GatherStagedSources.Count}";
    public string ButtonLogs => IsEnglish ? "Logs" : "Log";
    public string LogsOverlayTitle => IsEnglish ? "Runtime Logging Console" : "Console Logging Runtime";
    public string LogsOverlaySizeLabel => IsEnglish ? "Panel size" : "Dimensione pannello";
    public string LogsOverlayWidthLabel => IsEnglish ? "Width" : "Larghezza";
    public string LogsOverlayHeightLabel => IsEnglish ? "Height" : "Altezza";
    public string LogsOverlayClearButton => IsEnglish ? "Clear" : "Pulisci";
    public string LogsOverlayCloseButton => IsEnglish ? "Close" : "Chiudi";
    public string GatherTitleText => IsEnglish ? "Gather Dataset" : "Raccolta Dataset";
    public string GatherIntroText => IsEnglish
        ? "Acquire, convert, validate and handoff datasets before training."
        : "Ottieni, converti, valida e passa i dataset al training.";
    public string GatherSourceLabel => IsEnglish ? "Source (Provider URL or local path)" : "Sorgente (URL provider o percorso locale)";
    public string GatherSourceTooltipText => IsEnglish
        ? "Enter a provider dataset URL (Hugging Face/GitHub/HTTP) or a local file/folder path. For multiple sources, fetch one source at a time, then merge."
        : "Inserisci un URL dataset del provider (Hugging Face/GitHub/HTTP) oppure un percorso locale file/cartella. Per più sorgenti, fai fetch una alla volta e poi unisci.";
    public string GatherWorkspaceLabel => IsEnglish ? "Workspace" : "Workspace";
    public string GatherBrowseText => IsEnglish ? "Browse..." : "Sfoglia...";
    public string GatherWorkspaceButtonText => IsEnglish ? "Workspace..." : "Workspace...";
    public string GatherLicenseGateTitle => IsEnglish
        ? "License Gate (mandatory before fetch)"
        : "Controllo Licenza (obbligatorio prima del fetch)";
    public string GatherCheckLicenseButtonText => IsEnglish ? "Check License" : "Controlla Licenza";
    public string GatherStep1CheckLicenseText => $"1. {GatherCheckLicenseButtonText}";
    public string GatherLicenseAckText => IsEnglish
        ? "I verified license and confirm usage rights"
        : "Ho verificato la licenza e confermo i diritti d'uso";
    public string GatherFetchButtonText => IsEnglish ? "Fetch Source" : "Scarica Sorgente";
    public string GatherStep2FetchText => IsEnglish
        ? $"2. {GatherFetchButtonText} (repeat per source)"
        : $"2. {GatherFetchButtonText} (ripeti per ogni sorgente)";
    public string GatherConvertButtonText => IsEnglish ? "Convert Parquet" : "Converti Parquet";
    public string GatherStep3ConvertText => IsEnglish
        ? $"3. {GatherConvertButtonText} (optional, repeat)"
        : $"3. {GatherConvertButtonText} (opzionale, ripetibile)";
    public string GatherValidateButtonText => IsEnglish ? "Validate Dataset" : "Valida Dataset";
    public string GatherStep5ValidateText => $"5. {GatherValidateButtonText}";
    public string GatherApplyButtonText => IsEnglish ? "Apply Recommendations" : "Applica Raccomandazioni";
    public string GatherStep6ApplyText => $"6. {GatherApplyButtonText}";
    public string GatherMergeButtonText => IsEnglish ? "Merge Sources" : "Unisci Sorgenti";
    public string GatherStep4MergeText => $"4. {GatherMergeButtonText}";
    public string GatherClearButtonText => IsEnglish ? "Clear Dataset Staging" : "Pulisci Staging Dataset";
    public string GatherCancelButtonText => IsEnglish ? "Cancel" : "Annulla";
    public string GatherRemoveSourceButtonText => IsEnglish ? "Remove" : "Rimuovi";
    public string GatherHandoffScratchText => IsEnglish ? "Handoff to From-Scratch" : "Passa a From-Scratch";
    public string GatherStep7HandoffScratchText => $"7A. {GatherHandoffScratchText}";
    public string GatherHandoffFtText => IsEnglish ? "Handoff to Fine-Tuning" : "Passa a Fine-Tuning";
    public string GatherStep7HandoffFtText => $"7B. {GatherHandoffFtText}";
    public string GatherRecTokenizerText => IsEnglish ? "Recommended tokenizer:" : "Tokenizer consigliato:";
    public string GatherRecProfileText => IsEnglish ? "Recommended profile:" : "Profilo consigliato:";
    public string GatherMergeOptionsTitle => IsEnglish ? "Advanced Merge Options" : "Opzioni Merge Avanzate";
    public string GatherDedupPolicyLabel => IsEnglish ? "Dedup policy" : "Policy dedup";
    public string GatherSourcesTableTitle => IsEnglish ? "Staged sources (toggle + weight + license)" : "Sorgenti in staging (toggle + peso + licenza)";
    public string GatherFlowHintText => IsEnglish
        ? "Suggested order: 1) Check License -> 2) Fetch (repeat) -> 3) Convert Parquet if needed (repeat) -> 4) Merge -> 5) Validate -> 6) Apply -> 7) Handoff."
        : "Ordine consigliato: 1) Controlla Licenza -> 2) Scarica (ripeti) -> 3) Converti Parquet se serve (ripeti) -> 4) Unisci -> 5) Valida -> 6) Applica -> 7) Passa al flusso.";
    public string GatherRecommendedTokenizerLine => $"{GatherRecTokenizerText} {GatherRecommendedTokenizer}";
    public string GatherRecommendedTrainingProfileLine => $"{GatherRecProfileText} {GatherRecommendedTrainingProfile}";
    public string DatasetRecommendationsTitle => IsEnglish ? "Smart Recommendations" : "Raccomandazioni Smart";
    public string DatasetApplyRecommendationsText => IsEnglish ? "Apply Dataset Recommendations" : "Applica Raccomandazioni Dataset";
    public string DatasetRecommendedTokenizerLine => IsEnglish
        ? $"Recommended tokenizer: {DatasetRecommendedTokenizer}"
        : $"Tokenizer consigliato: {DatasetRecommendedTokenizer}";
    public string DatasetRecommendedTrainingProfileLine => IsEnglish
        ? $"Recommended training profile: {DatasetRecommendedTrainingProfile}"
        : $"Profilo training consigliato: {DatasetRecommendedTrainingProfile}";
    public string DatasetRecommendedTokenizer
    {
        get => _datasetRecommendedTokenizer;
        private set
        {
            if (!SetProperty(ref _datasetRecommendedTokenizer, value)) return;
            OnPropertyChanged(nameof(DatasetRecommendedTokenizerLine));
        }
    }
    public string DatasetRecommendedTrainingProfile
    {
        get => _datasetRecommendedTrainingProfile;
        private set
        {
            if (!SetProperty(ref _datasetRecommendedTrainingProfile, value)) return;
            OnPropertyChanged(nameof(DatasetRecommendedTrainingProfileLine));
        }
    }
    public string DatasetRecommendationsText
    {
        get => _datasetRecommendationsText;
        private set => SetProperty(ref _datasetRecommendationsText, value);
    }
    public string OllamaFtBaseModelPath
    {
        get => _ollamaFtBaseModelPath;
        set => SetProperty(ref _ollamaFtBaseModelPath, value);
    }
    public string OllamaFtOutputModelName
    {
        get => _ollamaFtOutputModelName;
        set => SetProperty(ref _ollamaFtOutputModelName, value);
    }
    public string OllamaFtDatasetPath
    {
        get => _ollamaFtDatasetPath;
        set => SetProperty(ref _ollamaFtDatasetPath, value);
    }
    public string OllamaFtOutputDirectory
    {
        get => _ollamaFtOutputDirectory;
        set => SetProperty(ref _ollamaFtOutputDirectory, value);
    }
    public string OllamaFtTemplate
    {
        get => _ollamaFtTemplate;
        set => SetProperty(ref _ollamaFtTemplate, value);
    }
    public string OllamaFtMethod
    {
        get => _ollamaFtMethod;
        set => SetProperty(ref _ollamaFtMethod, value);
    }
    public string OllamaFtBackend
    {
        get => _ollamaFtBackend;
        set => SetProperty(ref _ollamaFtBackend, value);
    }
    public int OllamaFtEpochs
    {
        get => _ollamaFtEpochs;
        set => SetProperty(ref _ollamaFtEpochs, value);
    }
    public int OllamaFtBatchSize
    {
        get => _ollamaFtBatchSize;
        set => SetProperty(ref _ollamaFtBatchSize, value);
    }
    public int OllamaFtGradientAccumulation
    {
        get => _ollamaFtGradientAccumulation;
        set => SetProperty(ref _ollamaFtGradientAccumulation, value);
    }
    public double OllamaFtLearningRate
    {
        get => _ollamaFtLearningRate;
        set => SetProperty(ref _ollamaFtLearningRate, value);
    }
    public int OllamaFtLoraRank
    {
        get => _ollamaFtLoraRank;
        set => SetProperty(ref _ollamaFtLoraRank, value);
    }
    public int OllamaFtLoraAlpha
    {
        get => _ollamaFtLoraAlpha;
        set => SetProperty(ref _ollamaFtLoraAlpha, value);
    }
    public double OllamaFtLoraDropout
    {
        get => _ollamaFtLoraDropout;
        set => SetProperty(ref _ollamaFtLoraDropout, value);
    }
    public bool OllamaFtPackForOllama
    {
        get => _ollamaFtPackForOllama;
        set => SetProperty(ref _ollamaFtPackForOllama, value);
    }
    public string OllamaFtStatusText
    {
        get => _ollamaFtStatusText;
        set => SetProperty(ref _ollamaFtStatusText, value);
    }
    public bool IsOllamaFtRunning
    {
        get => _isOllamaFtRunning;
        set
        {
            if (!SetProperty(ref _isOllamaFtRunning, value)) return;
            OnPropertyChanged(nameof(IsOllamaFtIdle));
        }
    }
    public bool IsOllamaFtIdle => !IsOllamaFtRunning;
    public bool CanConvertOllamaFineTuneToGguf
    {
        get
        {
            if (IsOllamaFtRunning) return false;
            var manifest = Path.Combine(OllamaFtOutputDirectory, "ollama_finetune_manifest.json");
            return File.Exists(manifest);
        }
    }
    public bool CanFinalizeOllamaFineTuneExport
    {
        get
        {
            if (IsOllamaFtRunning) return false;
            var gguf = Path.Combine(OllamaFtOutputDirectory, "exports", "ollama_finetune", "model.gguf");
            return File.Exists(gguf);
        }
    }
    public int OllamaFtPipelineStep
    {
        get
        {
            if (!CanConvertOllamaFineTuneToGguf) return 1;
            if (!CanFinalizeOllamaFineTuneExport) return 3;
            var exportsDir = Path.Combine(OllamaFtOutputDirectory, "exports", "ollama_finetune");
            var statusPath = Path.Combine(exportsDir, "ollama_handoff_status.json");
            if (!File.Exists(statusPath)) return 4;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(statusPath, Encoding.UTF8));
                if (doc.RootElement.TryGetProperty("status", out var s)
                    && string.Equals(s.GetString(), "ready", StringComparison.OrdinalIgnoreCase))
                    return 5;
            }
            catch
            {
                // ignore status parse errors
            }
            return 4;
        }
    }
    public double OllamaFtPipelineProgressPercent => Math.Clamp((OllamaFtPipelineStep - 1) / 4.0 * 100.0, 0.0, 100.0);
    public string OllamaFtPipelineProgressText => OllamaFtPipelineStep switch
    {
        1 => "Step 1/4: Prepare run",
        2 => "Step 2/4: Start fine-tune",
        3 => "Step 3/4: Convert to GGUF",
        4 => "Step 4/4: Finalize export",
        _ => "Pipeline completed"
    };
    public string ParameterEstimate => ParameterEstimator.Human(ParameterEstimator.Estimate(ModelConfig));
    public string GenerationExplanation => GenerationPreviewService.Describe(SamplingConfig);
    public string ChartSummary => BuildChartSummary();
    public string TokenizerRecommendationText => BuildTokenizerRecommendationText();
    public string TokenizerPresetBadgeText => BuildTokenizerPresetBadgeText();
    public string TokenizerIdealValuesText => BuildTokenizerIdealValuesText();
    public string RoadmapChecklistText => BuildRoadmapChecklistText();
    public string HardwareCpuText { get; private set; } = "Detecting CPU...";
    public string HardwareRamText { get; private set; } = "Detecting RAM...";
    public string HardwareOsText { get; private set; } = "Detecting OS...";
    public string HardwareBackendText { get; private set; } = "Detecting backend...";
    public string HardwareNotesText { get; private set; } = string.Empty;
    public string HardwareAutoDetectText => IsEnglish
        ? "Automatic startup detection: CPU, RAM, GPU and backend device used for training/generation."
        : "Rilevamento automatico all'avvio: CPU, RAM, GPU e device backend usato per training/generation.";
    public string TokenizationStepGuide => IsEnglish
        ? "Recommended order: 1) Train Tokenizer  2) Build x/y Preview."
        : "Ordine consigliato: 1) Train Tokenizer  2) Build x/y Preview.";
    public string TrainingStepGuide => IsEnglish
        ? "Recommended order: 1) Verify readiness is complete  2) Start Backend Training  3) Monitor logs/loss  4) Cancel only if needed  5) Open Output Folder for checkpoints."
        : "Ordine consigliato: 1) Verifica readiness completa  2) Start Backend Training  3) Monitora log/loss  4) Cancel solo se necessario  5) Open Output Folder per checkpoint.";
    public string TrainingWizardMiniText => IsEnglish
        ? "Training Wizard: 1) Select training profile  2) Select eval pack  3) Check preflight status  4) Start training."
        : "Training Wizard: 1) Seleziona profilo training  2) Seleziona eval pack  3) Controlla preflight  4) Avvia training.";
    public string TrainingWizardDetailedText => BuildTrainingWizardDetailedText();
    public string NotificationsTitleText => IsEnglish ? "Notifications" : "Notifiche";
    public string NotificationsButtonText => UnreadNotificationsCount > 0 ? $"🔔 {UnreadNotificationsCount}" : "🔔";
    public string NotificationsEmptyText => IsEnglish ? "No notifications yet." : "Nessuna notifica.";
    public string TrainingPreflightText => BuildTrainingPreflightText();
    public string DatasetHelpText => IsEnglish
        ? "You can import a single file or a folder of files (.txt/.md/.csv), or paste text manually."
        : "Puoi importare singolo file o cartella di file (.txt/.md/.csv), oppure incollare testo manualmente.";
    public string DatasetCopyPathText => IsEnglish ? "Copy Path" : "Copia Path";
    public string DatasetPathTooltipText => IsEnglish
        ? "Dataset source path. Scroll horizontally to inspect long paths."
        : "Percorso sorgente dataset. Scorri orizzontalmente per vedere i path lunghi.";
    public string DatasetStatsCharactersLabel => IsEnglish ? "Characters" : "Caratteri";
    public string DatasetStatsLinesLabel => IsEnglish ? "Lines" : "Righe";
    public string DatasetStatsWordsLabel => IsEnglish ? "Words (approx)" : "Parole (stima)";
    public string DatasetStatsUniqueCharsLabel => IsEnglish ? "Unique chars" : "Caratteri unici";
    public string DatasetPreviewLabelText => IsEnglish ? "Dataset Preview" : "Anteprima Dataset";
    public string DatasetPreviewHintText => IsEnglish
        ? "For very large datasets, this box shows a truncated preview only to keep the UI responsive. Full data remains on disk and is still used for processing/training."
        : "Per dataset molto grandi, questo riquadro mostra solo un'anteprima troncata per mantenere l'interfaccia reattiva. I dati completi restano su disco e vengono comunque usati per elaborazione/training.";
    public string RunDirectoryHelpText => IsEnglish
        ? "Output folder where JSONL logs, checkpoints and training manifest are saved."
        : "Cartella dove vengono salvati log JSONL, checkpoint e manifest del training.";
    public bool IsTrainingIdle => !IsTraining;
    public string TrainingStatusText
    {
        get => _trainingStatusText;
        private set => SetProperty(ref _trainingStatusText, value);
    }
    public string ClusterLiveSummaryText
    {
        get => _clusterLiveSummaryText;
        private set => SetProperty(ref _clusterLiveSummaryText, value);
    }
    public string ClusterLiveRoleText
    {
        get => _clusterLiveRoleText;
        private set => SetProperty(ref _clusterLiveRoleText, value);
    }
    public string ClusterLiveCoordinatorText
    {
        get => _clusterLiveCoordinatorText;
        private set => SetProperty(ref _clusterLiveCoordinatorText, value);
    }
    public bool TrainingForceCpu
    {
        get => TrainingConfig.ForceCpu;
        set
        {
            if (TrainingConfig.ForceCpu == value) return;
            TrainingConfig.ForceCpu = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string LastStepText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().Step.ToString("N0");
    public string LastTrainLossText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().TrainLoss.ToString("F6");
    public string LastValLossText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().ValLoss.ToString("F6");
    public string AvgTokensPerSecondText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Average(x => x.TokensPerSecond).ToString("F1");
    public string ElapsedTimeText
    {
        get
        {
            if (_trainingStartedAtUtc is null) return "-";
            var span = DateTimeOffset.UtcNow - _trainingStartedAtUtc.Value;
            if (span < TimeSpan.Zero) span = TimeSpan.Zero;
            return $"{(int)span.TotalMinutes:00}:{span.Seconds:00}";
        }
    }
    public string OverfittingStatusText => ComputeOverfittingStatus();
    public string TrainingQualityScoreText => ComputeTrainingQualityScore().score < 0 ? "-" : $"{ComputeTrainingQualityScore().score}/100";
    public string TrainingQualityColor => ComputeTrainingQualityScore().color;
    public string TrainingQualityBandText => ComputeTrainingQualityScore().band;
    public string TrainingQualityHintText => ComputeTrainingQualityScore().hint;
    public string EvalSuiteResultText => _lastEvalSummary is null
        ? "-"
        : $"{_lastEvalSummary.EvalSuite} ({_lastEvalSummary.NumBenchmarks} tasks)";
    public string EvalAverageScoreText => _lastEvalSummary is null ? "-" : _lastEvalSummary.AverageScore.ToString("F2");
    public string EvalBandText => _lastEvalSummary is null ? "-" : _lastEvalSummary.Band;
    public string EvalGateText => _lastEvalSummary is null ? "-" : (_lastEvalSummary.ReleaseGatePassed ? "PASS" : "FAIL");
    public string EvalGateThresholdText => _lastEvalSummary is null ? "-" : _lastEvalSummary.ReleaseGateThreshold.ToString("F1");
    public bool IsBackendBusy => _isBackendBootstrapping;
    public bool IsBackendConfigured => PythonBackendBridge.IsPythonAvailable(PythonPath);
    public bool ShowSetupBackendAction => !IsBackendConfigured;
    public string BackendStatusText => IsBackendBusy
        ? (_backendUserInitiatedStartup ? "Backend startup in progress..." : "Checking backend status...")
        : IsBackendConfigured
            ? $"Backend ready: {PythonPath}"
            : "Backend Python is not configured. Click 'Setup Backend' or set a valid interpreter path.";
    public double CpuUsage { get => _cpuUsage; set => SetProperty(ref _cpuUsage, value); }
    public double GpuUsage { get => _gpuUsage; set => SetProperty(ref _gpuUsage, value); }
    public double RamUsage { get => _ramUsage; set => SetProperty(ref _ramUsage, value); }
    public double DiskUsage { get => _diskUsage; set => SetProperty(ref _diskUsage, value); }
    public string CpuUsageText => $"{CpuUsage:0}%";
    public string GpuUsageText => $"{GpuUsage:0}%";
    public string RamUsageText => $"{RamUsage:0}%";
    public string DiskUsageText => $"{DiskUsage:0}%";
    public string WizardText => BuildWizardText();
    public string WizardProgressText => BuildWizardProgressText();
    public string WizardToggleText => IsEnglish ? "Guided Tour" : "Tour Guidato";
    public string SettingsTitle => IsEnglish ? "Settings" : "Impostazioni";
    public string AboutTitle => IsEnglish ? "About this software" : "Informazioni software";
    public string AboutText => IsEnglish
        ? "LLM Forge Studio\nDeveloped by Giovanni De Miccoli with OpenAI Codex support.\nDesktop application for local educational LLM experiments.\nVersion: 1.0.1\nLicense: MIT\nWebsite: www.giovannidemiccoli.it"
        : "LLM Forge Studio\nSviluppato da Giovanni De Miccoli con supporto OpenAI Codex.\nApplicazione desktop per esperimenti locali educativi su LLM.\nVersione: 1.0.1\nLicenza: MIT\nSito: www.giovannidemiccoli.it";
    public string LanguageLabel => IsEnglish ? "Language" : "Lingua";
    public string ThemeLabel => IsEnglish ? "Theme" : "Tema";
    public string LightLabel => IsEnglish ? "Light" : "Chiaro";
    public string DarkLabel => IsEnglish ? "Dark" : "Scuro";
    public string SectionWorkflowText => IsEnglish ? "Workflow" : "Workflow";
    public string HeaderSubtitleText => IsEnglish
        ? "Tokenizer, training loop, generation, and local LLM lab"
        : "Tokenizer, training loop, generation e laboratorio per LLM personali";
    public string ButtonLoadProject => IsEnglish ? "Load Project" : "Carica Progetto";
    public string ButtonSaveProject => IsEnglish ? "Save Project" : "Salva Progetto";
    public string ButtonGuide => IsEnglish ? "Guide" : "Guida";
    public string ButtonTrainTokenizer => IsEnglish ? "Train Tokenizer" : "Allena Tokenizer";
    public string WelcomeTitle => IsEnglish ? "Welcome to LLM Forge Studio" : "Benvenuto in LLM Forge Studio";
    public string GuideTitleText => IsEnglish ? "Guide - Page by Page" : "Guida - Pagina per Pagina";
    public string GuideIntroText => IsEnglish
        ? "Quick guide adapted from llm-from-scratch topics (excluding competition)."
        : "Guida rapida semplificata dalle parti 01-05 di llm-from-scratch (esclusa la competition).";
    public string GuideThanksText => IsEnglish
        ? "A special thank you to angelos-p for making this excellent guide available to everyone."
        : "Un ringraziamento speciale ad angelos-p per aver reso disponibile questa guida eccellente.";
    public string GuideDatasetText => IsEnglish
        ? "Goal: prepare text data. Import files/folder (.txt/.md/.csv), clean content, and validate dataset stats."
        : "Obiettivo: preparare il testo. Importa file o cartella (.txt/.md/.csv), pulisci e controlla le statistiche.";
    public string GuideTokenizationText => IsEnglish
        ? "Goal: convert text into token IDs. Run Train Tokenizer, then Build x/y Preview to validate next-token shift."
        : "Obiettivo: convertire testo in ID token. Prima Train Tokenizer, poi Build x/y Preview per verificare lo shift.";
    public string GuideTokenizationKeyText => IsEnglish
        ? "Key idea: smaller vocab on smaller datasets usually improves stability."
        : "Idea chiave: vocabolario piccolo su dataset piccolo evita sparsità e migliora la stabilità.";
    public string GuideModelText => IsEnglish
        ? "Goal: define Transformer capacity (block size, layers, heads, embedding) balancing quality and cost."
        : "Obiettivo: definire la capacità del Transformer bilanciando qualità e costi.";
    public string GuideModelKeyText => IsEnglish
        ? "Key idea: vocab size drives embeddings/output, while most params live in Transformer blocks."
        : "Idea chiave: il vocab size impatta embedding/output, mentre molti parametri sono nei blocchi Transformer.";
    public string GuideTrainingText => IsEnglish
        ? "Goal: learn next-token prediction. Start backend training and monitor train/val loss in real time."
        : "Obiettivo: apprendere next-token prediction. Avvia il backend training e monitora train/val loss.";
    public string GuideTrainingKeyText => IsEnglish
        ? "Key idea: if val loss rises while train loss improves, overfitting is likely."
        : "Idea chiave: se la val loss sale mentre train migliora, è probabile overfitting.";
    public string GuideGenerationText => IsEnglish
        ? "Goal: generate from checkpoint using prompt, temperature, top-k, and seed."
        : "Obiettivo: generare dal checkpoint usando prompt, temperature, top-k e seed.";
    public string GuideGenerationKeyText => IsEnglish
        ? "Key idea: evaluate quality frequently with real prompts, not metrics alone."
        : "Idea chiave: valuta spesso la qualità con prompt reali, non solo metriche.";
    public string ModelContextWindowText => IsEnglish
        ? $"Context Window (input tokens): {ModelBlockSize:N0}. This is the max token span the model can read per step."
        : $"Finestra di contesto (token input): {ModelBlockSize:N0}. È il massimo span di token leggibile per step.";
    public string GenerationOutputWindowText => IsEnglish
        ? $"Max Output Tokens: {SamplingConfig.MaxNewTokens:N0}. This is the maximum length of generated reply."
        : $"Token output massimi: {SamplingConfig.MaxNewTokens:N0}. È la lunghezza massima della risposta generata.";
    public string GuideOrderText => IsEnglish
        ? "Recommended workflow: Dataset -> Tokenization -> Model -> Training -> Generation."
        : "Ordine operativo consigliato: Dataset -> Tokenization -> Model -> Training -> Generation.";
    public string GuideSourceText => IsEnglish
        ? "Source: github.com/angelos-p/llm-from-scratch/docs."
        : "Fonte: github.com/angelos-p/llm-from-scratch/docs.";
    public bool ShowWizardOverlay => WizardEnabled && !ShowSettingsOverlay && !ShowAboutOverlay && !ShowLogsOverlay && !ShowNotificationsOverlay;
    public string HeaderBackground => IsLightTheme ? "#DDF1FF" : "#0D1C33";
    public string SidebarBackground => IsLightTheme ? "#CBE5FF" : "#0A1528";
    public string DividerBrush => IsLightTheme ? "#84AEDD" : "#5F8FC9";
    public string ModalBackground => IsLightTheme ? "#EBF7FF" : "#142946";
    public string ModalBorderBrush => IsLightTheme ? "#8EB5DF" : "#7DB2F3";

    public void ExportProjectToLog()
    {
        var payload = BuildProjectPayload();
        Log = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
    }

    public async Task SaveProjectAsync(string path)
    {
        var project = BuildProjectPayload();
        await ProjectStore.SaveAsync(project, path);
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "project.save",
            "Project saved from UI.",
            new { path, snapshot = project });
        Log = $"Project saved: {path}";
    }

    public async Task LoadProjectAsync(string path)
    {
        var project = await ProjectStore.LoadAsync(path);
        DatasetPath = project.DatasetPath;
        DatasetText = project.DatasetText;
        _datasetUsesExternalSource = project.DatasetUsesExternalSource;
        _datasetExternalSourcePath = project.DatasetExternalSourcePath ?? string.Empty;
        if (_datasetUsesExternalSource && !string.IsNullOrWhiteSpace(_datasetExternalSourcePath))
            _ = RefreshDatasetStatsFromSourceAsync(_datasetExternalSourcePath);
        ResetPreparationState();
        _wizardDatasetImported = !string.IsNullOrWhiteSpace(DatasetText);

        ApplyCleaner(project.Cleaner);
        ApplyTokenizer(project.Tokenizer);
        ApplyModel(project.Model);
        ApplyTraining(project.Training);
        var loadedProfile = string.IsNullOrWhiteSpace(project.SelectedTrainingProfile)
            ? InferTrainingProfileFromConfig(project.Training)
            : project.SelectedTrainingProfile;
        if (!TrainingProfileOptions.Contains(loadedProfile))
            loadedProfile = "Custom";
        SelectedTrainingProfile = loadedProfile;
        NormalizeLoadedTrainingProfileConsistency();
        ApplySampling(project.Sampling);

        SelectedSection = Sections.Contains(project.SelectedSection) ? project.SelectedSection : "Dataset";
        PythonPath = project.PythonPath;
        RunDirectory = project.RunDirectory;
        CheckpointPath = project.CheckpointPath;
        GenerationPrompt = project.GenerationPrompt;
        ApplyGatherProjectState(project.Gather);
        var wf = project.Workflow ?? new WorkflowProjectState();
        _wizardSetupDone = wf.WizardSetupDone;
        _wizardDatasetImported = wf.WizardDatasetImported || !string.IsNullOrWhiteSpace(DatasetText);
        _wizardTokenizerTrained = wf.WizardTokenizerTrained;
        _wizardPreviewBuilt = wf.WizardPreviewBuilt;
        _wizardTrainingStarted = wf.WizardTrainingStarted;
        _wizardCheckpointSet = wf.WizardCheckpointSet || !string.IsNullOrWhiteSpace(CheckpointPath);
        if (!string.IsNullOrWhiteSpace(wf.TokenizerStatusText))
            TokenizerStatusText = wf.TokenizerStatusText;
        if (!string.IsNullOrWhiteSpace(wf.BatchPreviewStatusText))
            BatchPreviewStatusText = wf.BatchPreviewStatusText;
        if (_wizardTokenizerTrained)
        {
            await TryRestoreTokenizationStateAsync();
            if (_lastTokenization is null)
            {
                _wizardTokenizerTrained = false;
                TokenizerStatusText = IsEnglish
                    ? "Tokenizer state file missing. Re-run tokenization."
                    : "File stato tokenizer mancante. Riesegui la tokenizzazione.";
                TokenizerProgressText = IsEnglish ? "Tokenizer idle." : "Tokenizer inattivo.";
                TokenizerLiveStatsText = IsEnglish ? "State not restored." : "Stato non ripristinato.";
            }
        }

        RefreshAll();
        OnPropertyChanged(nameof(IsTokenizerReady));
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "project.load",
            "Project loaded from UI.",
            new { path, snapshot = BuildProjectPayload() });
        Log = $"Project loaded: {path}";
    }

    public async Task ImportDatasetFromPathAsync(string path)
    {
        var cleaned = await DatasetLoader.LoadTextAsync(path, Cleaner);
        DatasetPath = path;
        ApplyDatasetUiPayload(cleaned, path);
        _ = RefreshDatasetStatsFromSourceAsync(path);
        ResetPreparationState();
        _wizardDatasetImported = true;
        Log = $"Dataset imported: {Path.GetFileName(path)}";
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    public async Task ImportDatasetFromFolderAsync(string folderPath)
    {
        if (!Directory.Exists(folderPath))
        {
            Log = $"Cartella non trovata: {folderPath}";
            return;
        }

        var files = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
            .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        if (files.Count == 0)
        {
            Log = "Nessun file supportato trovato nella cartella (.txt/.md/.csv/.jsonl/.json).";
            return;
        }

        DatasetPath = folderPath;
        var preview = await BuildFolderDatasetPreviewAsync(files);
        ApplyDatasetUiPayload(preview, folderPath, forceExternalMode: true);
        _ = RefreshDatasetStatsFromSourceAsync(folderPath);
        ResetPreparationState();
        _wizardDatasetImported = true;

        var rlhfCandidate = ResolveRlhfFeedbackPathCandidate(files);
        if (!string.IsNullOrWhiteSpace(rlhfCandidate))
        {
            TrainingConfig.RlhfFeedbackPath = rlhfCandidate;
            OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
            Log = $"Dataset cartella importato: {files.Count} file uniti.\nAuto-detect RLHF feedback path: {rlhfCandidate}";
        }
        else
        {
            Log = $"Dataset cartella importato: {files.Count} file uniti.";
        }

        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    private void ApplyDatasetUiPayload(string text, string sourcePath, bool forceExternalMode = false)
    {
        var effective = text ?? string.Empty;
        var external = forceExternalMode || effective.Length > DatasetUiPreviewLimitChars;
        _datasetUsesExternalSource = external;
        _datasetExternalSourcePath = external ? sourcePath : string.Empty;

        if (external)
        {
            var preview = effective.Length > DatasetUiPreviewLimitChars
                ? effective[..DatasetUiPreviewLimitChars]
                : effective;
            var suffix = IsEnglish
                ? $"\n\n[Large dataset mode] Full dataset remains on disk at:\n{sourcePath}\nUI preview is truncated to keep the app responsive."
                : $"\n\n[Modalità dataset grande] Il dataset completo resta su disco in:\n{sourcePath}\nL'anteprima UI è limitata per mantenere l'app reattiva.";
            DatasetText = preview + suffix;
        }
        else
        {
            DatasetText = effective;
        }
    }

    private async Task RefreshDatasetStatsFromSourceAsync(string sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath)) return;
        if (!_datasetUsesExternalSource) return;
        DatasetCharsText = IsEnglish ? "calculating..." : "calcolo...";
        DatasetLinesText = IsEnglish ? "calculating..." : "calcolo...";
        DatasetWordsText = IsEnglish ? "calculating..." : "calcolo...";
        DatasetUniqueCharsText = IsEnglish ? "calculating..." : "calcolo...";
        OnPropertyChanged(nameof(DatasetCharsText));
        OnPropertyChanged(nameof(DatasetLinesText));
        OnPropertyChanged(nameof(DatasetWordsText));
        OnPropertyChanged(nameof(DatasetUniqueCharsText));

        try
        {
            var stats = await Task.Run(() => ComputeExternalDatasetStats(sourcePath));
            DatasetCharsText = stats.chars.ToString("N0");
            DatasetLinesText = stats.lines.ToString("N0");
            DatasetWordsText = stats.words.ToString("N0");
            DatasetUniqueCharsText = stats.uniqueChars.ToString("N0");
            OnPropertyChanged(nameof(DatasetCharsText));
            OnPropertyChanged(nameof(DatasetLinesText));
            OnPropertyChanged(nameof(DatasetWordsText));
            OnPropertyChanged(nameof(DatasetUniqueCharsText));
            RecomputeDatasetRecommendations(stats.chars);
        }
        catch
        {
            DatasetCharsText = "-";
            DatasetLinesText = "-";
            DatasetWordsText = "-";
            DatasetUniqueCharsText = "-";
            OnPropertyChanged(nameof(DatasetCharsText));
            OnPropertyChanged(nameof(DatasetLinesText));
            OnPropertyChanged(nameof(DatasetWordsText));
            OnPropertyChanged(nameof(DatasetUniqueCharsText));
            RecomputeDatasetRecommendations(null);
        }
    }

    private static (long chars, long lines, long words, int uniqueChars) ComputeExternalDatasetStats(string sourcePath)
    {
        long chars = 0;
        long lines = 0;
        long words = 0;
        var unique = new HashSet<char>();

        static bool IsDataFile(string p) =>
            p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
            || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
            || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
            || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
            || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase);

        IEnumerable<string> files = File.Exists(sourcePath)
            ? new[] { sourcePath }
            : Directory.Exists(sourcePath)
                ? Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories).Where(IsDataFile)
                : Enumerable.Empty<string>();

        foreach (var file in files)
        {
            using var reader = new StreamReader(file);
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (line is null) continue;
                lines++;
                chars += line.Length + 1;
                words += line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
                foreach (var ch in line)
                    unique.Add(ch);
            }
        }

        return (chars, lines, words, unique.Count);
    }

    private async Task<string> BuildFolderDatasetPreviewAsync(IReadOnlyList<string> files)
    {
        var sb = new StringBuilder();
        var limit = Math.Min(files.Count, 6);
        for (var i = 0; i < limit; i++)
        {
            var cleaned = await DatasetLoader.LoadTextAsync(files[i], Cleaner);
            if (sb.Length > 0) sb.Append("\n\n");
            var take = cleaned.Length > 20_000 ? cleaned[..20_000] : cleaned;
            sb.Append(take);
            if (sb.Length >= DatasetUiPreviewLimitChars) break;
        }
        return sb.ToString();
    }

    private static string ResolveRlhfFeedbackPathCandidate(IReadOnlyList<string> files)
    {
        var jsonlFiles = files
            .Where(p => p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();
        if (jsonlFiles.Count == 0) return string.Empty;
        if (jsonlFiles.Count == 1) return jsonlFiles[0];

        var heuristic = jsonlFiles.FirstOrDefault(p =>
        {
            var name = Path.GetFileName(p).ToLowerInvariant();
            return name.Contains("rlhf")
                || name.Contains("feedback")
                || name.Contains("human");
        });
        return heuristic ?? string.Empty;
    }

    private void RefreshAll()
    {
        RefreshStats();
        RefreshWarnings();
        OnPropertyChanged(nameof(ParameterEstimate));
        OnPropertyChanged(nameof(GenerationExplanation));
        OnPropertyChanged(nameof(ModelContextWindowText));
        OnPropertyChanged(nameof(GenerationOutputWindowText));
        OnPropertyChanged(nameof(ChartSummary));
        OnPropertyChanged(nameof(TokenizerRecommendationText));
        OnPropertyChanged(nameof(BackendStatusText));
        OnPropertyChanged(nameof(IsBackendConfigured));
        OnPropertyChanged(nameof(ShowSetupBackendAction));
        OnPropertyChanged(nameof(RunDirectoryHelpText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    public async Task RefreshHardwareAsync()
    {
        var root = ResolveProjectRoot();
        var summary = await HardwareProbe.DetectAsync(root, PythonPath);
        _lastHardwareSummary = summary;
        HardwareCpuText = $"{summary.CpuName} ({summary.LogicalCores} logical cores)";
        HardwareRamText = summary.TotalRamGb > 0 ? $"{summary.TotalRamGb:F1} GB RAM" : "RAM non rilevata";
        HardwareOsText = summary.OsDescription;
        HardwareBackendText = summary.BackendDeviceStatus;
        HardwareNotesText = summary.BackendNotes;

        HardwareGpus.Clear();
        foreach (var gpu in summary.Gpus) HardwareGpus.Add(gpu);

        OnPropertyChanged(nameof(HardwareCpuText));
        OnPropertyChanged(nameof(HardwareRamText));
        OnPropertyChanged(nameof(HardwareOsText));
        OnPropertyChanged(nameof(HardwareBackendText));
        OnPropertyChanged(nameof(HardwareNotesText));
    }

    private void RefreshStats()
    {
        if (_datasetUsesExternalSource && !string.IsNullOrWhiteSpace(_datasetExternalSourcePath))
        {
            if (File.Exists(_datasetExternalSourcePath))
            {
                var fi = new FileInfo(_datasetExternalSourcePath);
                DatasetCharsText = "-";
                DatasetLinesText = "-";
                DatasetWordsText = "-";
                DatasetUniqueCharsText = "-";
                DatasetStatsText = IsEnglish
                    ? $"External dataset mode | file={Path.GetFileName(_datasetExternalSourcePath)} | size={fi.Length:N0} bytes"
                    : $"Modalità dataset esterno | file={Path.GetFileName(_datasetExternalSourcePath)} | dimensione={fi.Length:N0} byte";
                OnPropertyChanged(nameof(DatasetStatsText));
                OnPropertyChanged(nameof(DatasetCharsText));
                OnPropertyChanged(nameof(DatasetLinesText));
                OnPropertyChanged(nameof(DatasetWordsText));
                OnPropertyChanged(nameof(DatasetUniqueCharsText));
                RecomputeDatasetRecommendations(null);
                return;
            }
            if (Directory.Exists(_datasetExternalSourcePath))
            {
                var count = Directory.EnumerateFiles(_datasetExternalSourcePath, "*.*", SearchOption.AllDirectories).Count();
                DatasetCharsText = "-";
                DatasetLinesText = "-";
                DatasetWordsText = "-";
                DatasetUniqueCharsText = "-";
                DatasetStatsText = IsEnglish
                    ? $"External dataset mode | folder={_datasetExternalSourcePath} | files={count:N0}"
                    : $"Modalità dataset esterno | cartella={_datasetExternalSourcePath} | file={count:N0}";
                OnPropertyChanged(nameof(DatasetStatsText));
                OnPropertyChanged(nameof(DatasetCharsText));
                OnPropertyChanged(nameof(DatasetLinesText));
                OnPropertyChanged(nameof(DatasetWordsText));
                OnPropertyChanged(nameof(DatasetUniqueCharsText));
                RecomputeDatasetRecommendations(null);
                return;
            }
        }

        var stats = TextCleaner.Analyze(DatasetText);
        DatasetCharsText = stats.CharacterCount.ToString("N0");
        DatasetLinesText = stats.LineCount.ToString("N0");
        DatasetWordsText = stats.ApproxWordCount.ToString("N0");
        DatasetUniqueCharsText = stats.UniqueCharacterCount.ToString("N0");
        DatasetStatsText = $"Chars: {stats.CharacterCount:N0} | Lines: {stats.LineCount:N0} | Words: {stats.ApproxWordCount:N0} | Unique chars: {stats.UniqueCharacterCount:N0}";
        OnPropertyChanged(nameof(DatasetCharsText));
        OnPropertyChanged(nameof(DatasetLinesText));
        OnPropertyChanged(nameof(DatasetWordsText));
        OnPropertyChanged(nameof(DatasetUniqueCharsText));
        OnPropertyChanged(nameof(DatasetStatsText));
        RecomputeDatasetRecommendations(stats.CharacterCount);
    }

    private void RecomputeDatasetRecommendations(long? knownChars)
    {
        long chars = knownChars ?? 0;
        if (chars <= 0 && long.TryParse((DatasetCharsText ?? string.Empty).Replace(",", string.Empty), out var parsed))
            chars = parsed;
        if (chars <= 0) chars = Math.Max(0, DatasetText?.Length ?? 0);

        var tokenizer = chars > 2_000_000 ? "ByteLevelBpe" : "SimpleBpe";
        var profile = chars switch
        {
            < 200_000 => "Tiny",
            < 2_500_000 => "Balanced",
            < 8_000_000 => "Serious",
            _ => "Research"
        };

        DatasetRecommendedTokenizer = tokenizer;
        DatasetRecommendedTrainingProfile = profile;
        DatasetRecommendationsText = IsEnglish
            ? $"Auto-analysis suggests `{tokenizer}` tokenizer and `{profile}` training profile based on dataset scale ({chars:N0} chars)."
            : $"L'analisi automatica suggerisce tokenizer `{tokenizer}` e profilo training `{profile}` in base alla scala dataset ({chars:N0} caratteri).";
    }

    private void ApplyDatasetRecommendations()
    {
        if (string.Equals(DatasetRecommendedTokenizer, "ByteLevelBpe", StringComparison.OrdinalIgnoreCase))
            SelectedTokenizerKind = TokenizerKind.ByteLevelBpe;
        else
            SelectedTokenizerKind = TokenizerKind.SimpleBpe;

        SelectedTrainingProfile = string.IsNullOrWhiteSpace(DatasetRecommendedTrainingProfile)
            ? "Balanced"
            : DatasetRecommendedTrainingProfile;
        GuidedDefaultsEngine.ApplyTrainingProfile(SelectedTrainingProfile, TrainingConfig);
        ApplyRecommendedModelAndTrainingSettings(SelectedTokenizerKind);

        Log = IsEnglish
            ? $"Dataset recommendations applied: tokenizer={SelectedTokenizerKind}, profile={SelectedTrainingProfile}."
            : $"Raccomandazioni dataset applicate: tokenizer={SelectedTokenizerKind}, profilo={SelectedTrainingProfile}.";
        RefreshWarnings();
        OnPropertyChanged(nameof(TrainingReadinessText));
    }

    private async Task TrainTokenizerAsync()
    {
        if (IsTokenizerBusy) return;
        _tokenizerCts?.Cancel();
        _tokenizerCts?.Dispose();
        _tokenizerCts = new CancellationTokenSource();
        var cancellationToken = _tokenizerCts.Token;
        IsTokenizerBusy = true;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            TokenizerProgressValue = 10;
            TokenizerProgressText = IsEnglish ? "Loading dataset..." : "Caricamento dataset...";
            TokenizerLiveStatsText = IsEnglish ? "Initializing tokenizer runtime telemetry..." : "Inizializzazione telemetria runtime tokenizer...";
            _lastTokenizerRuntimeLogAtUtc = DateTimeOffset.MinValue;

            var datasetPayload = await ResolveDatasetTextForProcessingAsync();
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(datasetPayload))
            {
                Log = "Cannot train tokenizer: dataset text is empty.";
                TokenizerProgressValue = 0;
                TokenizerProgressText = IsEnglish ? "Tokenizer idle." : "Tokenizer inattivo.";
                return;
            }
            if (LooksLikeMetadataOnlyDataset(datasetPayload))
            {
                Log = IsEnglish
                    ? "Tokenizer blocked: staged dataset appears to contain mostly metadata/schema (README/card/info) instead of real training rows. Re-run Gather: fetch data files -> convert parquet if needed -> merge -> validate."
                    : "Tokenizer bloccato: il dataset sembra contenere soprattutto metadata/schema (README/card/info) invece di righe reali di training. Riesegui Gather: fetch file dati -> converti parquet se necessario -> merge -> valida.";
                TokenizerProgressValue = 0;
                TokenizerProgressText = IsEnglish ? "Tokenizer idle." : "Tokenizer inattivo.";
                TokenizerLiveStatsText = IsEnglish ? "Blocked by dataset quality gate." : "Bloccato dal controllo qualità dataset.";
                return;
            }

            TokenizerProgressValue = 35;
            var estimatedSeconds = EstimateTokenizerSeconds(datasetPayload.Length);
            var trainingPhaseText = IsEnglish ? "Training tokenizer..." : "Training tokenizer...";
            TokenizerProgressText = BuildTokenizerEtaText(trainingPhaseText, estimatedSeconds);
            var trainingStopwatch = Stopwatch.StartNew();
            using var estimatorCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var estimatorTask = RunTokenizerProgressEstimatorAsync(
                estimatorCts.Token,
                trainingStopwatch,
                estimatedSeconds,
                35,
                84,
                trainingPhaseText,
                datasetPayload.Length,
                IsEnglish ? "chars" : "caratteri");
            var tokenizerTask = Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokenizer = TokenizerRegistry.Create(SelectedTokenizerKind);
                tokenizer.Train(datasetPayload, TokenizerConfig);
                var ids = tokenizer.Encode(datasetPayload);
                const int decodePreviewTokenLimit = 8192;
                var decodePreviewIds = ids.Count > decodePreviewTokenLimit
                    ? ids.Take(decodePreviewTokenLimit).ToList()
                    : ids;
                var decoded = tokenizer.Decode(decodePreviewIds);
                return new
                {
                    tokenizer.Name,
                    Vocabulary = tokenizer.Vocabulary.ToList(),
                    TokenIds = ids,
                    Decoded = decoded,
                    DecodedSampled = ids.Count > decodePreviewTokenLimit,
                    DecodedTokenCount = decodePreviewIds.Count
                };
            });
            var tokenizerOutcome = await AwaitOrCancelAsync(tokenizerTask, cancellationToken);
            estimatorCts.Cancel();
            await SafeAwaitEstimatorAsync(estimatorTask);
            cancellationToken.ThrowIfCancellationRequested();
            trainingStopwatch.Stop();
            UpdateTokenizerThroughput(SelectedTokenizerKind, datasetPayload.Length, trainingStopwatch.Elapsed.TotalSeconds);
            var tokPerSec = tokenizerOutcome.TokenIds.Count / Math.Max(0.001, trainingStopwatch.Elapsed.TotalSeconds);
            TokenizerLiveStatsText = IsEnglish
                ? $"Done | tokens: {tokenizerOutcome.TokenIds.Count:N0} | elapsed: {trainingStopwatch.Elapsed:mm\\:ss} | avg: {tokPerSec:N0} tok/s"
                : $"Completato | token: {tokenizerOutcome.TokenIds.Count:N0} | tempo: {trainingStopwatch.Elapsed:mm\\:ss} | media: {tokPerSec:N0} tok/s";

            TokenizerProgressValue = 85;
            TokenizerProgressText = IsEnglish ? "Updating preview..." : "Aggiornamento anteprima...";
            _lastTokenization = new TokenizationResult
            {
                TokenIds = tokenizerOutcome.TokenIds,
                Vocabulary = tokenizerOutcome.Vocabulary,
                DecodedPreview = tokenizerOutcome.Decoded
            };
            await SaveTokenizationStateAsync();
            _wizardTokenizerTrained = true;

            VocabularyPreview.Clear();
            foreach (var item in tokenizerOutcome.Vocabulary.Take(200))
                VocabularyPreview.Add(item);

            ModelConfig.VocabSize = Math.Max(1, tokenizerOutcome.Vocabulary.Count);
            var decodedPreview = TruncateForUi(tokenizerOutcome.Decoded, TokenizerUiLogLimitChars, IsEnglish
                ? "[Tokenizer decoded preview truncated for UI performance.]"
                : "[Anteprima tokenizer troncata per performance UI.]");
            var previewScopeNote = tokenizerOutcome.DecodedSampled
                ? IsEnglish
                    ? $"Decoded preview uses a fast sample of first {tokenizerOutcome.DecodedTokenCount:N0} tokens (full decode skipped for performance on large datasets)."
                    : $"L'anteprima decodificata usa un campione veloce dei primi {tokenizerOutcome.DecodedTokenCount:N0} token (decode completo saltato per performance su dataset grandi)."
                : IsEnglish
                    ? $"Decoded preview uses full token sequence ({tokenizerOutcome.DecodedTokenCount:N0} tokens)."
                    : $"L'anteprima decodificata usa la sequenza completa ({tokenizerOutcome.DecodedTokenCount:N0} token).";
            Log = $"Tokenizer trained: {tokenizerOutcome.Name}. Vocab={tokenizerOutcome.Vocabulary.Count}, Tokens={tokenizerOutcome.TokenIds.Count}.\n{previewScopeNote}\nDecoded preview:\n{decodedPreview}";
            TokenizerStatusText = $"Tokenizer ready: {tokenizerOutcome.Name}, vocab {tokenizerOutcome.Vocabulary.Count:N0}, tokens {tokenizerOutcome.TokenIds.Count:N0}.";
            BatchPreviewStatusText = "x/y preview not built yet.";
            RaiseModelTrainingBindingsChanged();
            RefreshWarnings();
            OnPropertyChanged(nameof(ParameterEstimate));
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
            OnPropertyChanged(nameof(IsTokenizerReady));
            OnPropertyChanged(nameof(IsTrainingReady));
            OnPropertyChanged(nameof(TrainingReadinessText));

            TokenizerProgressValue = 100;
            TokenizerProgressText = IsEnglish ? "Tokenizer training completed." : "Training tokenizer completato.";
            await Task.Delay(120);
        }
        catch (OperationCanceledException)
        {
            TokenizerProgressText = IsEnglish ? "Tokenizer operation canceled." : "Operazione tokenizer annullata.";
            Log = IsEnglish ? "Tokenizer run canceled." : "Esecuzione tokenizer annullata.";
            TokenizerLiveStatsText = IsEnglish ? "Canceled by user." : "Annullato dall'utente.";
        }
        finally
        {
            TokenizerProgressValue = 0;
            IsTokenizerBusy = false;
            _tokenizerCts?.Dispose();
            _tokenizerCts = null;
            if (!string.Equals(TokenizerProgressText, IsEnglish ? "Tokenizer operation canceled." : "Operazione tokenizer annullata.", StringComparison.Ordinal))
            {
                TokenizerProgressText = IsEnglish ? "Tokenizer idle." : "Tokenizer inattivo.";
                if (TokenizerLiveStatsText == "-")
                    TokenizerLiveStatsText = IsEnglish ? "Idle." : "Inattivo.";
            }
        }
    }

    private async Task BuildBatchPreviewAsync()
    {
        if (IsTokenizerBusy) return;
        if (_lastTokenization is null)
            await TrainTokenizerAsync();
        if (_lastTokenization is null) return;
        _tokenizerCts?.Cancel();
        _tokenizerCts?.Dispose();
        _tokenizerCts = new CancellationTokenSource();
        var cancellationToken = _tokenizerCts.Token;
        IsTokenizerBusy = true;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            TokenizerProgressValue = 20;
            var estimatedSeconds = EstimatePreviewSeconds(_lastTokenization.TokenIds.Count, ModelConfig.BlockSize);
            var previewPhaseText = IsEnglish ? "Building x/y preview..." : "Costruzione anteprima x/y...";
            TokenizerProgressText = BuildTokenizerEtaText(previewPhaseText, estimatedSeconds);
            var previewStopwatch = Stopwatch.StartNew();
            using var estimatorCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var estimatorTask = RunTokenizerProgressEstimatorAsync(
                estimatorCts.Token,
                previewStopwatch,
                estimatedSeconds,
                20,
                84,
                previewPhaseText,
                _lastTokenization.TokenIds.Count,
                IsEnglish ? "tokens" : "token");

            var tokenIds = _lastTokenization.TokenIds;
            var previewTask = Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return TrainingBatchBuilder.BuildPreview(tokenIds, ModelConfig.BlockSize);
            });
            var preview = await AwaitOrCancelAsync(previewTask, cancellationToken);
            estimatorCts.Cancel();
            await SafeAwaitEstimatorAsync(estimatorTask);
            cancellationToken.ThrowIfCancellationRequested();
            previewStopwatch.Stop();
            _wizardPreviewBuilt = true;
            var tokPerSec = _lastTokenization.TokenIds.Count / Math.Max(0.001, previewStopwatch.Elapsed.TotalSeconds);
            TokenizerLiveStatsText = IsEnglish
                ? $"Preview done | source tokens: {_lastTokenization.TokenIds.Count:N0} | elapsed: {previewStopwatch.Elapsed:mm\\:ss} | avg: {tokPerSec:N0} tok/s"
                : $"Anteprima completata | token sorgente: {_lastTokenization.TokenIds.Count:N0} | tempo: {previewStopwatch.Elapsed:mm\\:ss} | media: {tokPerSec:N0} tok/s";

            TokenizerProgressValue = 85;
            var logText = preview.Explanation + "\n\nX:\n" + string.Join(", ", preview.X.Take(64)) + "\n\nY:\n" + string.Join(", ", preview.Y.Take(64));
            Log = TruncateForUi(logText, TokenizerUiLogLimitChars, IsEnglish
                ? "[x/y preview output truncated for UI performance.]"
                : "[Output anteprima x/y troncato per performance UI.]");
            BatchPreviewStatusText = $"x/y preview ready: X={preview.X.Count:N0}, Y={preview.Y.Count:N0}, showing first 64 IDs.";
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
            OnPropertyChanged(nameof(IsBatchPreviewReady));
            OnPropertyChanged(nameof(IsTrainingReady));
            OnPropertyChanged(nameof(TrainingReadinessText));
            TokenizerProgressValue = 100;
            TokenizerProgressText = IsEnglish ? "x/y preview completed." : "Anteprima x/y completata.";
            await Task.Delay(120);
        }
        catch (OperationCanceledException)
        {
            TokenizerProgressText = IsEnglish ? "Tokenizer operation canceled." : "Operazione tokenizer annullata.";
            Log = IsEnglish ? "x/y preview canceled." : "Anteprima x/y annullata.";
            TokenizerLiveStatsText = IsEnglish ? "Canceled by user." : "Annullato dall'utente.";
        }
        finally
        {
            TokenizerProgressValue = 0;
            IsTokenizerBusy = false;
            _tokenizerCts?.Dispose();
            _tokenizerCts = null;
            if (!string.Equals(TokenizerProgressText, IsEnglish ? "Tokenizer operation canceled." : "Operazione tokenizer annullata.", StringComparison.Ordinal))
                TokenizerProgressText = IsEnglish ? "Tokenizer idle." : "Tokenizer inattivo.";
        }
    }
    private async Task<string> WriteConsolidatedDatasetForBackendAsync()
    {
        var raw = await ResolveDatasetTextForProcessingAsync();
        if (string.IsNullOrWhiteSpace(raw))
        {
            // Fallback 1: try DatasetPath explicitly (file or folder import root)
            raw = await TryLoadDatasetFromPathAsync(DatasetPath);
        }
        if (string.IsNullOrWhiteSpace(raw))
        {
            // Fallback 2: if Gather produced a merged dataset, use it.
            raw = await TryLoadDatasetFromPathAsync(GatherStagedDatasetPath);
        }
        if (string.IsNullOrWhiteSpace(raw))
        {
            throw new InvalidOperationException(
                IsEnglish
                    ? "Training blocked: resolved dataset is empty before backend launch. Re-import dataset or rerun Gather merge/validate."
                    : "Training bloccato: dataset risolto vuoto prima dell'avvio backend. Reimporta il dataset o riesegui merge/validate in Gather.");
        }

        var quality = AnalyzeTrainingCorpusQuality(raw);
        if (!quality.IsAcceptable)
        {
            var msg = IsEnglish
                ? $"Training blocked by quality gate: {quality.Message}"
                : $"Training bloccato dal quality gate: {quality.Message}";
            throw new InvalidOperationException(msg);
        }
        else if (quality.Message.StartsWith("warning", StringComparison.OrdinalIgnoreCase)
                 || quality.Message.StartsWith("avviso", StringComparison.OrdinalIgnoreCase))
        {
            Log = IsEnglish
                ? $"Quality gate warning: {quality.Message}"
                : $"Avviso quality gate: {quality.Message}";
            AddNotification("warning",
                IsEnglish ? "Dataset quality warning" : "Avviso qualità dataset",
                quality.Message);
        }

        // Auto-chat safeguard: if dataset looks like prompt/response or messages JSONL and
        // no alignment was selected, use SFT formatting so generation behaves like chat reply
        // instead of raw JSON completion.
        if (!TrainingConfig.FineTuningOrchestration
            && string.Equals(TrainingConfig.AlignmentMode, "none", StringComparison.OrdinalIgnoreCase)
            && LooksLikeStructuredChatDataset(raw))
        {
            TrainingConfig.AlignmentMode = "sft";
            OnPropertyChanged(nameof(TrainingAlignmentMode));
            Log = IsEnglish
                ? "Auto-alignment enabled: detected structured prompt/response dataset -> using SFT formatting for chat behavior."
                : "Allineamento automatico attivato: rilevato dataset strutturato prompt/response -> uso formato SFT per comportamento chat.";
        }
        return await FineTuningStageOrchestrator.PrepareAsync(raw, TrainingConfig, RunDirectory);
    }

    private sealed record CorpusQualityGateResult(bool IsAcceptable, string Message);

    private CorpusQualityGateResult AnalyzeTrainingCorpusQuality(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
            return new CorpusQualityGateResult(false, IsEnglish ? "empty dataset content." : "contenuto dataset vuoto.");

        var lines = raw.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(x => x.Trim())
            .Where(x => x.Length > 0)
            .ToList();

        if (lines.Count < 300)
        {
            return new CorpusQualityGateResult(
                false,
                IsEnglish
                    ? "too few non-empty lines (<300). Dataset is too small for stable training."
                    : "troppo poche righe non vuote (<300). Dataset troppo piccolo per training stabile.");
        }

        var structured = TryAnalyzeStructuredJsonlQuality(lines);
        if (structured is not null)
            return structured;

        var meaningful = lines
            .Select(NormalizeQualityLine)
            .Where(x => x.Length >= 24)
            .ToList();
        if (meaningful.Count < 300)
            meaningful = lines;

        var unique = new HashSet<string>(meaningful, StringComparer.Ordinal);
        var uniqueRatio = unique.Count / (double)Math.Max(1, meaningful.Count);
        if (uniqueRatio < 0.10)
        {
            return new CorpusQualityGateResult(
                false,
                IsEnglish
                    ? $"low line uniqueness ({uniqueRatio:P0}). High repetition detected: likely template collapse."
                    : $"unicità righe bassa ({uniqueRatio:P0}). Rilevata alta ripetizione: probabile template collapse.");
        }
        if (uniqueRatio < 0.35)
        {
            return new CorpusQualityGateResult(
                true,
                IsEnglish
                    ? $"warning: line uniqueness is modest ({uniqueRatio:P0}). Training allowed, but response diversity may be limited."
                    : $"avviso: unicità righe moderata ({uniqueRatio:P0}). Training consentito, ma la diversità delle risposte potrebbe essere limitata.");
        }

        var templateMarkers = new[]
        {
            "ti rispondo in modo semplice",
            "ecco una guida pratica",
            "controlla a fine giornata",
            "misura i progressi una volta al giorno"
        };
        var lowered = raw.ToLowerInvariant();
        var templateHits = templateMarkers.Sum(m => CountOccurrences(lowered, m));
        var templateDensity = templateHits / Math.Max(1.0, lines.Count);
        if (templateDensity > 0.30)
        {
            return new CorpusQualityGateResult(
                false,
                IsEnglish
                    ? "template phrase density too high. Reduce repeated boilerplate before training."
                    : "densità frasi template troppo alta. Riduci boilerplate ripetuto prima del training.");
        }
        if (templateDensity > 0.18)
        {
            return new CorpusQualityGateResult(
                true,
                IsEnglish
                    ? "warning: template phrase density is elevated. Training allowed, but style repetition may increase."
                    : "avviso: densità frasi template elevata. Training consentito, ma potrebbe aumentare la ripetizione stilistica.");
        }

        return new CorpusQualityGateResult(true, IsEnglish ? "quality gate passed." : "quality gate superato.");
    }

    private static string NormalizeQualityLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line)) return string.Empty;
        var s = line.Trim().ToLowerInvariant();
        s = s.Replace('\t', ' ');

        // Remove common list/formatting prefixes that otherwise fake low uniqueness.
        if (s.Length >= 3 && char.IsDigit(s[0]) && (s[1] == '.' || s[1] == ')' || s[1] == ':'))
            s = s[2..].TrimStart();
        if (s.StartsWith("- ", StringComparison.Ordinal) || s.StartsWith("* ", StringComparison.Ordinal))
            s = s[2..].TrimStart();

        // Collapse whitespace.
        while (s.Contains("  ", StringComparison.Ordinal))
            s = s.Replace("  ", " ", StringComparison.Ordinal);

        return s;
    }

    private CorpusQualityGateResult? TryAnalyzeStructuredJsonlQuality(List<string> lines)
    {
        var records = new List<(string Prompt, string Response)>();
        var checkedLines = 0;
        foreach (var line in lines)
        {
            if (++checkedLines > 30_000) break;
            if (!line.StartsWith("{", StringComparison.Ordinal)) continue;
            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                if (root.TryGetProperty("prompt", out var p) && root.TryGetProperty("response", out var r))
                {
                    var prompt = (p.GetString() ?? string.Empty).Trim();
                    var response = (r.GetString() ?? string.Empty).Trim();
                    if (prompt.Length > 0 && response.Length > 0)
                        records.Add((prompt, response));
                }
            }
            catch
            {
                // non-JSONL line or partial write; ignore
            }
        }

        if (records.Count < 200)
            return null;

        var uniquePairs = records
            .Select(x => x.Prompt + "\n" + x.Response)
            .Distinct(StringComparer.Ordinal)
            .Count();
        var pairUniq = uniquePairs / (double)records.Count;

        var uniquePrompts = records.Select(x => x.Prompt).Distinct(StringComparer.Ordinal).Count();
        var promptUniq = uniquePrompts / (double)records.Count;

        if (pairUniq < 0.35 || promptUniq < 0.25)
        {
            var msg = IsEnglish
                ? $"low structured uniqueness (pairs={pairUniq:P0}, prompts={promptUniq:P0}). Likely template collapse."
                : $"unicità strutturata bassa (coppie={pairUniq:P0}, prompt={promptUniq:P0}). Probabile template collapse.";
            return new CorpusQualityGateResult(false, msg);
        }
        if (pairUniq < 0.60 || promptUniq < 0.45)
        {
            var warn = IsEnglish
                ? $"warning: structured uniqueness is moderate (pairs={pairUniq:P0}, prompts={promptUniq:P0}). Training allowed."
                : $"avviso: unicità strutturata moderata (coppie={pairUniq:P0}, prompt={promptUniq:P0}). Training consentito.";
            return new CorpusQualityGateResult(true, warn);
        }

        return new CorpusQualityGateResult(true, IsEnglish ? "quality gate passed (structured JSONL)." : "quality gate superato (JSONL strutturato).");
    }

    private static int CountOccurrences(string text, string token)
    {
        if (string.IsNullOrWhiteSpace(text) || string.IsNullOrWhiteSpace(token)) return 0;
        var count = 0;
        var index = 0;
        while (true)
        {
            index = text.IndexOf(token, index, StringComparison.Ordinal);
            if (index < 0) break;
            count++;
            index += token.Length;
        }

        return count;
    }

    private static bool LooksLikeStructuredChatDataset(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return false;
        var sample = raw.Length > 80_000 ? raw[..80_000] : raw;
        var lines = sample.Split('\n', StringSplitOptions.RemoveEmptyEntries).Take(120);
        var score = 0;
        foreach (var line in lines)
        {
            var t = line.Trim();
            if (!t.StartsWith("{")) continue;
            if (t.Contains("\"prompt\"", StringComparison.OrdinalIgnoreCase) && t.Contains("\"response\"", StringComparison.OrdinalIgnoreCase))
                score += 3;
            if (t.Contains("\"messages\"", StringComparison.OrdinalIgnoreCase))
                score += 2;
            if (t.Contains("\"role\"", StringComparison.OrdinalIgnoreCase) && t.Contains("\"content\"", StringComparison.OrdinalIgnoreCase))
                score += 2;
            if (score >= 6) return true;
        }
        return false;
    }

    private async Task<string> TryLoadDatasetFromPathAsync(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath)) return string.Empty;
        if (File.Exists(sourcePath))
        {
            var text = await DatasetLoader.LoadTextAsync(sourcePath, Cleaner);
            return text ?? string.Empty;
        }
        if (Directory.Exists(sourcePath))
        {
            var files = Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories)
                .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                         || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                         || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
                         || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
                         || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                .OrderBy(p => p)
                .ToList();
            if (files.Count == 0) return string.Empty;
            var blocks = new List<string>(files.Count);
            foreach (var file in files)
            {
                var text = await DatasetLoader.LoadTextAsync(file, Cleaner);
                if (!string.IsNullOrWhiteSpace(text))
                    blocks.Add(text);
            }
            return string.Join("\n\n", blocks);
        }
        return string.Empty;
    }

    private async Task<string> ResolveDatasetTextForProcessingAsync()
    {
        var fallback = DatasetText ?? string.Empty;
        if (_datasetUsesExternalSource && !string.IsNullOrWhiteSpace(_datasetExternalSourcePath))
        {
            if (File.Exists(_datasetExternalSourcePath))
            {
                var loaded = await DatasetLoader.LoadTextAsync(_datasetExternalSourcePath, Cleaner);
                if (!string.IsNullOrWhiteSpace(loaded)) return loaded;
                return fallback;
            }

            if (Directory.Exists(_datasetExternalSourcePath))
            {
                var files = Directory.GetFiles(_datasetExternalSourcePath, "*.*", SearchOption.AllDirectories)
                    .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                             || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                             || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
                             || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
                             || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                    .OrderBy(p => p)
                    .ToList();
                if (files.Count == 0) return fallback;
                var blocks = new List<string>(files.Count);
                foreach (var file in files)
                {
                    var loaded = await DatasetLoader.LoadTextAsync(file, Cleaner);
                    if (!string.IsNullOrWhiteSpace(loaded))
                        blocks.Add(loaded);
                }
                var merged = string.Join("\n\n", blocks);
                return string.IsNullOrWhiteSpace(merged) ? fallback : merged;
            }
        }

        return fallback;
    }

    private async Task PrepareRlhfFeedbackForRunAsync()
    {
        if (!TrainingConfig.FineTuneStageRlhf) return;
        if (!string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)) return;
        if (RlhfCollectedFeedback.Count == 0) return;

        var path = Path.Combine(RunDirectory, "rlhf_feedback_collected.jsonl");
        await RlhfFeedbackCollector.SaveJsonlAsync(RlhfCollectedFeedback, path);
        TrainingConfig.RlhfFeedbackPath = path;
        OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
    }

    private void AddRlhfFeedback()
    {
        var prompt = (RlhfDraftPrompt ?? string.Empty).Trim();
        var chosen = (RlhfDraftChosen ?? string.Empty).Trim();
        var rejected = (RlhfDraftRejected ?? string.Empty).Trim();
        if (prompt.Length == 0 || chosen.Length == 0)
        {
            Log = "RLHF feedback requires prompt and chosen response.";
            return;
        }

        RlhfCollectedFeedback.Add(new RlhfFeedbackRecord
        {
            Prompt = prompt,
            Chosen = chosen,
            Rejected = rejected
        });

        RlhfDraftPrompt = string.Empty;
        RlhfDraftChosen = string.Empty;
        RlhfDraftRejected = string.Empty;
        OnPropertyChanged(nameof(RlhfCollectedCountText));
    }

    private void ClearRlhfFeedback()
    {
        RlhfCollectedFeedback.Clear();
        OnPropertyChanged(nameof(RlhfCollectedCountText));
    }

    private async Task SetupBackendAsync()
    {
        ShowStartupOverlay = false;
        await EnsureBackendReadyAsync(forceInstall: false, userInitiated: true);
        await RefreshHardwareAsync();
    }


    private async Task ExportForOllamaAsync()
    {
        if (IsTraining)
        {
            Log = "Export Ollama non disponibile durante il training.";
            return;
        }
        if (!TrainingConfig.ExportGguf)
        {
            Log = "Abilita prima Export GGUF per usare Export for Ollama.";
            return;
        }

        var result = OllamaExportPackager.CreateFromRun(RunDirectory);
        TrainingStatusText = result.Status == "ready"
            ? "Ollama export completed."
            : "Ollama export blocked.";
        Log = $"Ollama export ({result.Status}): {result.ExportDirectory}\n{result.Notes}";
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.export.ollama.manual",
            "Manual Ollama export invoked from UI.",
            new
            {
                runDirectory = RunDirectory,
                status = result.Status,
                exportDirectory = result.ExportDirectory,
                notes = result.Notes
            });
        OnPropertyChanged(nameof(CanExportForOllama));
        OnPropertyChanged(nameof(OllamaExportButtonHint));
    }

    private void AutoSelectCheckpointIfAvailable()
    {
        var manifest = Path.Combine(RunDirectory, "checkpoint_manifest.json");
        if (!File.Exists(manifest)) return;
        CheckpointPath = manifest;
        _wizardCheckpointSet = true;
    }

    private string ComputeOverfittingStatus()
    {
        if (TrainingLogs.Count < 4) return "Overfitting signal: not enough validation points yet.";

        var valSeries = TrainingLogs.Select(x => x.ValLoss).ToList();
        var trainSeries = TrainingLogs.Select(x => x.TrainLoss).ToList();
        var bestVal = valSeries.Min();
        var currentVal = valSeries[^1];
        var previousVal = valSeries[^2];
        var currentTrain = trainSeries[^1];
        var previousTrain = trainSeries[^2];

        var valWorsening = currentVal > previousVal && currentVal > bestVal * 1.10;
        var trainImproving = currentTrain <= previousTrain;
        if (valWorsening && trainImproving)
            return "Overfitting signal: warning (val loss worsening while train loss improves).";

        if (currentVal <= bestVal * 1.02)
            return "Overfitting signal: healthy (validation near best).";

        return "Overfitting signal: monitor (no strong overfitting pattern yet).";
    }

    private (int score, string color, string band, string hint) ComputeTrainingQualityScore()
    {
        if (TrainingLogs.Count < 4)
            return (-1, "#94A3B8", "Not enough data", "Run longer training and collect more validation checkpoints.");

        var logs = TrainingLogs.ToList();
        var first = logs.First();
        var last = logs.Last();
        var bestVal = logs.Min(x => x.ValLoss);
        var valGain = first.ValLoss - last.ValLoss;
        var trainValGap = Math.Abs(last.TrainLoss - last.ValLoss);
        var overfitPenalty = last.ValLoss > bestVal * 1.08 && last.TrainLoss < last.ValLoss ? 18 : 0;
        var gainScore = Math.Clamp((int)Math.Round(valGain * 40.0), 0, 45);
        var stabilityScore = Math.Clamp((int)Math.Round((1.0 - Math.Min(trainValGap, 1.0)) * 35.0), 0, 35);
        var speedScore = Math.Clamp((int)Math.Round(Math.Min(logs.Average(x => x.TokensPerSecond) / 200000.0, 1.0) * 20.0), 0, 20);
        var score = Math.Clamp(gainScore + stabilityScore + speedScore - overfitPenalty, 0, 100);

        var band = "Poor";
        var color = "#EF4444";
        if (score >= 80) { band = "Excellent"; color = "#22C55E"; }
        else if (score >= 60) { band = "Good"; color = "#84CC16"; }
        else if (score >= 40) { band = "Fair"; color = "#F59E0B"; }

        var hint = score switch
        {
            < 40 => "Low score: increase dataset size/quality, reduce learning rate, and raise eval frequency.",
            < 60 => "Fair score: try lower learning rate and longer training with more regular validation checks.",
            < 80 => "Good score: tune block size and layers gradually, then compare validation trend.",
            _ => "Excellent score: keep settings stable and validate with new prompts/checkpoints."
        };

        return (score, color, band, hint);
    }

    private void RaiseTrainingDashboardChanged()
    {
        OnPropertyChanged(nameof(LastStepText));
        OnPropertyChanged(nameof(LastTrainLossText));
        OnPropertyChanged(nameof(LastValLossText));
        OnPropertyChanged(nameof(AvgTokensPerSecondText));
        OnPropertyChanged(nameof(ElapsedTimeText));
        OnPropertyChanged(nameof(OverfittingStatusText));
        OnPropertyChanged(nameof(TrainingQualityScoreText));
        OnPropertyChanged(nameof(TrainingQualityColor));
        OnPropertyChanged(nameof(TrainingQualityBandText));
        OnPropertyChanged(nameof(TrainingQualityHintText));
        OnPropertyChanged(nameof(EvalSuiteResultText));
        OnPropertyChanged(nameof(EvalAverageScoreText));
        OnPropertyChanged(nameof(EvalBandText));
        OnPropertyChanged(nameof(EvalGateText));
        OnPropertyChanged(nameof(EvalGateThresholdText));
    }

    private async Task RefreshEvalSummaryAsync()
    {
        var snapshot = await EvalSummaryReader.TryReadAsync(RunDirectory);
        await Dispatcher.UIThread.InvokeAsync(() =>
        {
            _lastEvalSummary = snapshot;
            RaiseTrainingDashboardChanged();
        });
    }

    private static async Task RunProcessAsync(string fileName, string arguments, string workingDirectory)
    {
        var info = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = Process.Start(info);
        if (process is null) throw new InvalidOperationException($"Impossibile avviare processo: {fileName}");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(error) ? output : error);
    }

    private async Task BootstrapBackendOnStartupAsync()
    {
        await EnsureBackendReadyAsync(forceInstall: false, userInitiated: false);
        await RefreshHardwareAsync();
        ShowStartupOverlay = false;
    }

    private async Task EnsureBackendReadyAsync(bool forceInstall, bool userInitiated)
    {
        if (_isBackendBootstrapping)
        {
            Log = IsEnglish ? "Backend setup is already running. Please wait." : "Setup backend già in corso, attendi il completamento...";
            return;
        }
        _isBackendBootstrapping = true;
        _backendUserInitiatedStartup = userInitiated;
        BackendSetupProgress = 0;
        BackendSetupStage = userInitiated ? "Backend startup in progress..." : "Checking backend status...";
        OnPropertyChanged(nameof(IsBackendBusy));
        OnPropertyChanged(nameof(BackendStatusText));
        try
        {
            var root = ResolveProjectRoot();
            var backendDir = Path.Combine(root, "backends", "python");
            if (!Directory.Exists(backendDir))
            {
                Log = IsEnglish ? $"Backend directory not found: {backendDir}" : $"Backend directory non trovata: {backendDir}";
                return;
            }

            var venvPython = OperatingSystem.IsWindows()
                ? Path.Combine(backendDir, ".venv", "Scripts", "python.exe")
                : Path.Combine(backendDir, ".venv", "bin", "python");

            var bundledPython = OperatingSystem.IsWindows()
                ? Path.Combine(root, "runtime", "python", "python.exe")
                : Path.Combine(root, "runtime", "python", "bin", "python3");

            string? bootstrapPython = null;
            if (!forceInstall && PythonBackendBridge.IsPythonAvailable(venvPython))
            {
                PythonPath = venvPython;
                _wizardSetupDone = true;
                SetBackendProgress(100, "Backend ready");
                Log = IsEnglish ? $"Backend ready (local venv): {venvPython}" : $"Backend pronto (venv locale): {venvPython}";
                return;
            }

            if (PythonBackendBridge.IsPythonAvailable(bundledPython))
                bootstrapPython = bundledPython;
            else if (PythonBackendBridge.IsPythonAvailable("python"))
                bootstrapPython = "python";
            else if (PythonBackendBridge.IsPythonAvailable("py"))
                bootstrapPython = "py";

            if (bootstrapPython is null)
            {
                SetBackendProgress(0, "Backend not configured");
                Log = IsEnglish
                    ? "Python was not found. Add embedded runtime in runtime/python or install system Python."
                    : "Python non trovato. Inserisci runtime embedded in runtime/python oppure installa Python di sistema.";
                return;
            }

            if (!File.Exists(venvPython))
            {
                SetBackendProgress(10, "Creating virtual environment...");
                Log = IsEnglish ? "Automatic backend setup: creating virtual environment..." : "Configurazione backend automatica: creazione virtualenv...";
                await RunProcessAsync(bootstrapPython, "-m venv .venv", backendDir);
            }
            SetBackendProgress(25, "Installing dependencies...");
            Log = IsEnglish ? "Automatic backend setup: installing dependencies..." : "Configurazione backend automatica: installazione dipendenze...";
            await InstallBackendDependenciesAsync(venvPython, backendDir);
            PythonPath = venvPython;
            _wizardSetupDone = true;
            SetBackendProgress(95, "Verifying backend...");
            await RunProcessAsync(venvPython, "-c \"import torch; print('ok')\"", backendDir);
            SetBackendProgress(100, "Backend ready");
            Log = IsEnglish ? $"Backend configured automatically: {venvPython}" : $"Backend configurato automaticamente: {venvPython}";
        }
        catch (Exception ex)
        {
            SetBackendProgress(0, IsEnglish ? "Setup failed" : "Setup fallito");
            Log = IsEnglish ? $"Backend bootstrap failed: {ex.Message}" : $"Bootstrap backend fallito: {ex.Message}";
        }
        finally
        {
            _isBackendBootstrapping = false;
            _backendUserInitiatedStartup = false;
            OnPropertyChanged(nameof(BackendStatusText));
            OnPropertyChanged(nameof(IsBackendConfigured));
            OnPropertyChanged(nameof(ShowSetupBackendAction));
            OnPropertyChanged(nameof(IsBackendBusy));
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
        }
    }

    private async Task InstallBackendDependenciesAsync(string venvPython, string backendDir)
    {
        var flavor = DetectBackendFlavor();

        SetBackendProgress(35, IsEnglish ? "Updating pip..." : "Aggiornamento pip...");
        await RunProcessAsync(venvPython, "-m pip install --upgrade pip", backendDir);
        SetBackendProgress(45, IsEnglish ? "Installing requirements..." : "Installazione requirements...");
        await RunProcessAsync(venvPython, "-m pip install -r requirements.txt", backendDir);
        SetBackendProgress(55, IsEnglish ? "Cleaning old torch packages..." : "Pulizia vecchi pacchetti torch...");
        await RunProcessAsync(venvPython, "-m pip uninstall -y torch torchvision torchaudio torch-directml", backendDir);

        switch (flavor)
        {
            case BackendFlavor.NvidiaCuda:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch CUDA (NVIDIA)..." : "Installazione PyTorch CUDA (NVIDIA)...");
                Log = IsEnglish ? "NVIDIA GPU detected: installing PyTorch CUDA..." : "GPU NVIDIA rilevata: installo PyTorch CUDA...";
                await RunProcessAsync(
                    venvPython,
                    "-m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
                    backendDir);
                break;
            case BackendFlavor.AmdDirectMl:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch (AMD)..." : "Installazione PyTorch (AMD)...");
                Log = IsEnglish ? "AMD GPU detected: installing PyTorch + DirectML..." : "GPU AMD rilevata: installo PyTorch + DirectML...";
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch torchvision torchaudio", backendDir);
                SetBackendProgress(82, IsEnglish ? "Installing DirectML..." : "Installazione DirectML...");
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch-directml", backendDir);
                break;
            default:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch CPU..." : "Installazione PyTorch CPU...");
                Log = IsEnglish ? "No compatible GPU detected: installing PyTorch CPU fallback..." : "Nessuna GPU compatibile rilevata: installo PyTorch CPU fallback...";
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch torchvision torchaudio", backendDir);
                break;
        }
        SetBackendProgress(90, IsEnglish ? "Finalizing..." : "Finalizzazione...");
    }

    private static BackendFlavor DetectBackendFlavor()
    {
        try
        {
            if (!OperatingSystem.IsWindows()) return BackendFlavor.Cpu;

            var info = new ProcessStartInfo
            {
                FileName = "cmd",
                Arguments = "/c wmic path win32_VideoController get Name",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var process = Process.Start(info);
            if (process is null) return BackendFlavor.Cpu;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);

            if (output.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase)) return BackendFlavor.NvidiaCuda;
            if (output.Contains("AMD", StringComparison.OrdinalIgnoreCase) || output.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                return BackendFlavor.AmdDirectMl;
        }
        catch
        {
            // ignored, fallback to CPU
        }

        return BackendFlavor.Cpu;
    }

    private async Task RefreshPerformanceAsync()
    {
        if (_isPerfSampling) return;
        _isPerfSampling = true;
        try
        {
            var values = await Task.Run(() => (
                cpu: ReadCpuUsage(),
                gpu: ReadGpuUsage(),
                ram: ReadRamUsage(),
                disk: ReadDiskActivityUsage()
            ));

            CpuUsage = values.cpu;
            GpuUsage = values.gpu;
            RamUsage = values.ram;
            DiskUsage = values.disk;
            OnPropertyChanged(nameof(CpuUsageText));
            OnPropertyChanged(nameof(GpuUsageText));
            OnPropertyChanged(nameof(RamUsageText));
            OnPropertyChanged(nameof(DiskUsageText));
        }
        finally
        {
            _isPerfSampling = false;
        }
    }

    private void RefreshClusterLiveStatus()
    {
        try
        {
            var runDir = string.IsNullOrWhiteSpace(RunDirectory) ? Path.Combine(Environment.CurrentDirectory, "runs", "default") : RunDirectory;
            var statePath = Path.Combine(runDir, "cluster_run_state.json");
            var hbPath = Path.Combine(runDir, "cluster_heartbeat.json");
            var role = (Environment.GetEnvironmentVariable("LLMFORGE_CLUSTER_ROLE") ?? "standalone").Trim().ToLowerInvariant();
            var nodeId = (Environment.GetEnvironmentVariable("LLMFORGE_CLUSTER_NODE_ID") ?? Environment.MachineName ?? "node-unknown").Trim();
            ClusterLiveRoleText = $"Role: {role} ({nodeId})";

            string orchestrator = string.Empty;
            string stateStatus = "idle";
            string heartbeatStatus = "idle";
            string ticket = string.Empty;
            if (File.Exists(statePath))
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(statePath, Encoding.UTF8));
                var root = doc.RootElement;
                stateStatus = root.TryGetProperty("status", out var s) ? s.GetString() ?? stateStatus : stateStatus;
                ticket = root.TryGetProperty("ticket", out var t) ? t.GetString() ?? string.Empty : string.Empty;
            }
            if (File.Exists(hbPath))
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(hbPath, Encoding.UTF8));
                var root = doc.RootElement;
                heartbeatStatus = root.TryGetProperty("status", out var s) ? s.GetString() ?? heartbeatStatus : heartbeatStatus;
                if (string.IsNullOrWhiteSpace(ticket))
                    ticket = root.TryGetProperty("ticket", out var t) ? t.GetString() ?? string.Empty : string.Empty;
            }

            var sharedRoot = Environment.GetEnvironmentVariable("LLMFORGE_CLUSTER_SHARED_ROOT");
            if (string.IsNullOrWhiteSpace(sharedRoot))
                sharedRoot = Path.Combine(runDir, "_cluster_shared");
            var queueRoot = Path.Combine(sharedRoot, "queue");
            var heartbeatsDir = Path.Combine(queueRoot, "heartbeats");
            var claimedDir = Path.Combine(queueRoot, "claimed");
            var pendingDir = Path.Combine(queueRoot, "pending");
            var resultDir = Path.Combine(queueRoot, "result");
            orchestrator = TrainingConfig.ClusterOrchestrator;

            ClusterLiveNodes.Clear();
            ClusterLiveRemoteGpus.Clear();
            var heartbeatFiles = Directory.Exists(heartbeatsDir)
                ? Directory.GetFiles(heartbeatsDir, "*.json", SearchOption.TopDirectoryOnly)
                : Array.Empty<string>();
            foreach (var hbFile in heartbeatFiles.OrderByDescending(f => File.GetLastWriteTimeUtc(f)))
            {
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(hbFile, Encoding.UTF8));
                    var r = doc.RootElement;
                    var nRole = r.TryGetProperty("role", out var xRole) ? xRole.GetString() ?? "-" : "-";
                    var nId = r.TryGetProperty("node_id", out var xNode) ? xNode.GetString() ?? Path.GetFileNameWithoutExtension(hbFile) : Path.GetFileNameWithoutExtension(hbFile);
                    var nStatus = r.TryGetProperty("status", out var xStatus) ? xStatus.GetString() ?? "alive" : "alive";
                    var nTicket = r.TryGetProperty("ticket", out var xTicket) ? xTicket.GetString() ?? string.Empty : string.Empty;
                    var updated = r.TryGetProperty("updated_at_utc", out var xUpd) ? xUpd.GetString() ?? string.Empty : string.Empty;
                    ClusterLiveNodes.Add($"{nId} | role={nRole} | status={nStatus} | ticket={nTicket} | updated={updated}");

                    if (r.TryGetProperty("gpu_devices", out var gpus) && gpus.ValueKind == JsonValueKind.Array)
                    {
                        var list = gpus.EnumerateArray().Select(e => e.GetString() ?? string.Empty).Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
                        if (list.Count > 0)
                            ClusterLiveRemoteGpus.Add($"{nId}: {string.Join(" | ", list)}");
                    }
                }
                catch
                {
                    ClusterLiveNodes.Add($"{Path.GetFileName(hbFile)} | parse-error");
                }
            }

            var pendingCount = Directory.Exists(pendingDir) ? Directory.GetFiles(pendingDir, "*.json").Length : 0;
            var claimedCount = Directory.Exists(claimedDir) ? Directory.GetFiles(claimedDir, "*.json").Length : 0;
            var resultCount = Directory.Exists(resultDir) ? Directory.GetFiles(resultDir, "*.json").Length : 0;
            if (ClusterLiveNodes.Count == 0)
                ClusterLiveNodes.Add("No live worker/coordinator heartbeat files detected yet.");
            if (ClusterLiveRemoteGpus.Count == 0)
                ClusterLiveRemoteGpus.Add("No remote GPU telemetry available yet.");

            ClusterLiveCoordinatorText = string.IsNullOrWhiteSpace(ticket)
                ? $"Coordinator link: {orchestrator} | shared root: {sharedRoot}"
                : $"Coordinator link: {orchestrator} | ticket={ticket} | shared root: {sharedRoot}";
            ClusterLiveSummaryText = $"Cluster live status: state={stateStatus}, heartbeat={heartbeatStatus}, pending={pendingCount}, claimed={claimedCount}, result={resultCount}.";
        }
        catch (Exception ex)
        {
            ClusterLiveSummaryText = $"Cluster live status unavailable: {ex.Message}";
            if (ClusterLiveNodes.Count == 0) ClusterLiveNodes.Add("Cluster monitor error.");
            if (ClusterLiveRemoteGpus.Count == 0) ClusterLiveRemoteGpus.Add("Cluster monitor error.");
        }
    }

    private static double ReadCpuUsage()
    {
        var output = RunCommand("cmd", "/c wmic cpu get loadpercentage /value");
        var line = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("LoadPercentage=", StringComparison.OrdinalIgnoreCase));
        if (line is null) return 0;
        return double.TryParse(line.Split('=', 2)[1].Trim(), out var value) ? Math.Clamp(value, 0, 100) : 0;
    }

    private static double ReadGpuUsage()
    {
        var output = RunCommand("cmd", "/c nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits");
        var line = output.Split('\n').Select(x => x.Trim()).FirstOrDefault(x => !string.IsNullOrWhiteSpace(x));
        if (line is null) return 0;
        if (double.TryParse(line, out var value)) return Math.Clamp(value, 0, 100);
        return 0;
    }

    private static double ReadRamUsage()
    {
        var output = RunCommand("cmd", "/c wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value");
        var freeLine = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("FreePhysicalMemory=", StringComparison.OrdinalIgnoreCase));
        var totalLine = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("TotalVisibleMemorySize=", StringComparison.OrdinalIgnoreCase));
        if (freeLine is null || totalLine is null) return 0;
        if (!double.TryParse(freeLine.Split('=', 2)[1].Trim(), out var freeKb)) return 0;
        if (!double.TryParse(totalLine.Split('=', 2)[1].Trim(), out var totalKb)) return 0;
        var used = Math.Max(0, totalKb - freeKb);
        return Math.Clamp(used / Math.Max(1, totalKb) * 100.0, 0, 100);
    }

    private static double ReadDiskUsage()
    {
        var root = Path.GetPathRoot(Directory.GetCurrentDirectory()) ?? "C:\\";
        var drive = new DriveInfo(root);
        var used = drive.TotalSize - drive.TotalFreeSpace;
        return Math.Clamp((double)used / Math.Max(1, drive.TotalSize) * 100.0, 0, 100);
    }

    private double ReadDiskActivityUsage()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                // Instantaneous disk throughput from Windows perf counters.
                var output = RunCommand("cmd", "/c wmic path Win32_PerfFormattedData_PerfDisk_LogicalDisk where \"Name='_Total'\" get DiskBytesPersec /value");
                var line = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("DiskBytesPersec=", StringComparison.OrdinalIgnoreCase));
                if (line is not null && long.TryParse(line.Split('=', 2)[1].Trim(), out var bytesPerSec))
                {
                    // Normalize with pragmatic saturation cap (300 MB/s => 100%).
                    return Math.Clamp(bytesPerSec / (300d * 1024d * 1024d) * 100d, 0, 100);
                }
                return 0;
            }

            // Linux fallback: derive activity from /proc/diskstats cumulative IO bytes.
            var diskstats = "/proc/diskstats";
            if (!File.Exists(diskstats)) return 0;
            var totalBytes = 0L;
            foreach (var line in File.ReadAllLines(diskstats))
            {
                var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 14) continue;
                var dev = parts[2];
                if (dev.StartsWith("loop", StringComparison.OrdinalIgnoreCase)) continue;
                if (dev.StartsWith("ram", StringComparison.OrdinalIgnoreCase)) continue;
                if (!long.TryParse(parts[5], out var sectorsRead)) continue;
                if (!long.TryParse(parts[9], out var sectorsWritten)) continue;
                totalBytes += (sectorsRead + sectorsWritten) * 512L;
            }

            var now = DateTimeOffset.UtcNow;
            if (_lastDiskIoSampleAtUtc == DateTimeOffset.MinValue || _lastDiskIoBytes <= 0 || totalBytes < _lastDiskIoBytes)
            {
                _lastDiskIoSampleAtUtc = now;
                _lastDiskIoBytes = totalBytes;
                return 0;
            }

            var elapsed = Math.Max(0.1, (now - _lastDiskIoSampleAtUtc).TotalSeconds);
            var delta = Math.Max(0, totalBytes - _lastDiskIoBytes);
            _lastDiskIoSampleAtUtc = now;
            _lastDiskIoBytes = totalBytes;
            var ioBytesPerSec = delta / elapsed;
            return Math.Clamp(ioBytesPerSec / (300d * 1024d * 1024d) * 100d, 0, 100);
        }
        catch
        {
            return 0;
        }
    }

    private string BuildWizardText()
    {
        if (!WizardEnabled) return IsEnglish ? "Guided tour disabled." : "Tour guidato disattivato.";
        if (!_wizardSetupDone)
            return IsEnglish
                ? "Step 1/6 - Hardware: click Setup Backend. Why: the app installs the right runtime (NVIDIA CUDA / AMD DirectML / CPU fallback) so training can run."
                : "Step 1/6 - Hardware: clicca Setup Backend. Perché: l'app installa il runtime corretto (NVIDIA CUDA / AMD DirectML / CPU fallback) per poter fare training.";

        if (!_wizardDatasetImported)
            return IsEnglish
                ? "Step 2/6 - Dataset: import file/folder first. Why: model quality depends directly on data quality and size."
                : "Step 2/6 - Dataset: importa prima file/cartella. Perché: qualità e dimensione del dato determinano la qualità del modello.";

        if (!_wizardTokenizerTrained)
            return IsEnglish
                ? "Step 3/6 - Tokenization: run Train Tokenizer first. Why: it builds the vocabulary and converts text into token IDs."
                : "Step 3/6 - Tokenization: esegui prima Train Tokenizer. Perché: costruisce il vocabolario e converte il testo in token ID.";

        if (!_wizardPreviewBuilt && !_wizardTrainingStarted)
            return IsEnglish
                ? "Step 4/6 - Tokenization check (optional): Build x/y Preview to verify next-token shift (Y is X shifted by +1) before training."
                : "Step 4/6 - Controllo tokenization (opzionale): Build x/y Preview per verificare lo shift next-token (Y è X spostato di +1) prima del training.";

        if (!_wizardTrainingStarted)
            return IsEnglish
                ? "Step 5/6 - Training: click Start Backend Training. Why: this creates checkpoint artifacts used by generation."
                : "Step 5/6 - Training: clicca Start Backend Training. Perché: crea i file checkpoint necessari alla generazione.";

        if (!_wizardCheckpointSet)
            return IsEnglish
                ? "Step 6/6 - Generation setup: set checkpoint_manifest.json path in Generation."
                : "Step 6/6 - Setup generation: imposta il path di checkpoint_manifest.json in Generation.";

        return IsEnglish
            ? "Step 6/6 - Generate: run Generate from Checkpoint, then iterate temperature/top-k/seed. Why: this is where you evaluate coherence and style."
            : "Step 6/6 - Generate: esegui Generate from Checkpoint, poi itera temperature/top-k/seed. Perché: qui valuti coerenza e stile del modello.";
    }

    private string BuildWizardProgressText()
    {
        if (!WizardEnabled) return IsEnglish ? "Tour off" : "Tour off";
        if (!_wizardSetupDone) return "1/6";
        if (!_wizardDatasetImported) return "2/6";
        if (!_wizardTokenizerTrained) return "3/6";
        if (!_wizardPreviewBuilt && !_wizardTrainingStarted) return "4/6";
        if (!_wizardTrainingStarted) return "5/6";
        if (!_wizardCheckpointSet) return "6/6";
        return "6/6";
    }

    private void SetBackendProgress(int value, string stage)
    {
        BackendSetupProgress = Math.Clamp(value, 0, 100);
        BackendSetupStage = stage;
        Log = $"[{BackendSetupProgress}%] {stage}";
    }

    private void RefreshWarnings()
    {
        Warnings.Clear();
        var datasetChars = DatasetText.Length;
        var tokenCount = _lastTokenization?.TokenCount ?? 0;
        foreach (var w in TokenizerCompatibility.Validate(TokenizerConfig, datasetChars, ModelConfig.BlockSize)) Warnings.Add(w);
        foreach (var w in CompatibilityRules.Validate(ModelConfig, TrainingConfig, tokenCount)) Warnings.Add(w);

        if (TrainingConfig.DistributedTraining)
        {
            var worldSizeRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
            var hasWorldSize = int.TryParse(worldSizeRaw, out var worldSize) && worldSize > 1;
            if (!hasWorldSize)
                Warnings.Add("Distributed mode is enabled but WORLD_SIZE is not set (>1). Run may fallback to single-device behavior.");
        }

        if (TrainingConfig.EnablePostTrainingQuantization && !TrainingConfig.ForceCpu)
            Warnings.Add("Post-training quantization currently applies on CPU path. Enable Force CPU if you want quantized artifact generation.");

        if (TrainingConfig.MixedPrecision && TrainingConfig.ForceCpu)
            Warnings.Add("Mixed precision is enabled with Force CPU. AMP acceleration is typically effective only on CUDA devices.");

        if (!Warnings.Any()) Warnings.Add("No major compatibility warnings.");
    }

    private ForgeProject BuildProjectPayload() => new()
    {
        DatasetPath = DatasetPath,
        DatasetText = DatasetText,
        DatasetUsesExternalSource = _datasetUsesExternalSource,
        DatasetExternalSourcePath = _datasetExternalSourcePath,
        Cleaner = Clone(Cleaner),
        Tokenizer = Clone(TokenizerConfig),
        Model = Clone(ModelConfig),
        Training = Clone(TrainingConfig),
        SelectedTrainingProfile = SelectedTrainingProfile,
        Sampling = Clone(SamplingConfig),
        SelectedSection = SelectedSection,
        PythonPath = PythonPath,
        RunDirectory = RunDirectory,
        CheckpointPath = CheckpointPath,
        GenerationPrompt = GenerationPrompt,
        Gather = new GatherProjectState
        {
            SourceInput = GatherSourceInput,
            WorkspaceDirectory = GatherWorkspaceDirectory,
            StagedDatasetPath = GatherStagedDatasetPath,
            ValidationText = GatherValidationText,
            RecommendedTokenizer = GatherRecommendedTokenizer,
            RecommendedTrainingProfile = GatherRecommendedTrainingProfile,
            MergeComplianceText = GatherMergeComplianceText,
            DedupPolicy = GatherDedupPolicy,
            StagedSources = GatherStagedSources.ToList(),
            SourceEntries = GatherSourceEntries.Select(x => new GatherSourceProjectState
            {
                Path = x.Path,
                Provider = x.Provider,
                LicenseLabel = x.LicenseLabel,
                IsLicensePermitted = x.IsLicensePermitted,
                IsEnabled = x.IsEnabled,
                Weight = x.Weight
            }).ToList()
        },
        Workflow = new WorkflowProjectState
        {
            WizardSetupDone = _wizardSetupDone,
            WizardDatasetImported = _wizardDatasetImported,
            WizardTokenizerTrained = _wizardTokenizerTrained,
            WizardPreviewBuilt = _wizardPreviewBuilt,
            WizardTrainingStarted = _wizardTrainingStarted,
            WizardCheckpointSet = _wizardCheckpointSet,
            TokenizerStatusText = TokenizerStatusText,
            BatchPreviewStatusText = BatchPreviewStatusText
        }
    };

    private void ApplyGatherProjectState(GatherProjectState? state)
    {
        GatherStagedSources.Clear();
        GatherSourceEntries.Clear();

        if (state is null)
        {
            OnPropertyChanged(nameof(GatherSourcesCountText));
            OnPropertyChanged(nameof(CanGatherConvertParquet));
            OnPropertyChanged(nameof(GatherNeedsParquetConversion));
            OnPropertyChanged(nameof(GatherParquetHintText));
            OnPropertyChanged(nameof(CanGatherMergeSources));
            OnPropertyChanged(nameof(CanGatherValidateDataset));
            return;
        }

        GatherSourceInput = state.SourceInput ?? string.Empty;
        GatherWorkspaceDirectory = state.WorkspaceDirectory ?? string.Empty;
        GatherStagedDatasetPath = state.StagedDatasetPath ?? string.Empty;
        GatherValidationText = string.IsNullOrWhiteSpace(state.ValidationText) ? "-" : state.ValidationText;
        GatherRecommendedTokenizer = string.IsNullOrWhiteSpace(state.RecommendedTokenizer) ? "-" : state.RecommendedTokenizer;
        GatherRecommendedTrainingProfile = string.IsNullOrWhiteSpace(state.RecommendedTrainingProfile) ? "-" : state.RecommendedTrainingProfile;
        GatherMergeComplianceText = string.IsNullOrWhiteSpace(state.MergeComplianceText)
            ? (IsEnglish ? "Merge compliance: pending" : "Compliance merge: in attesa")
            : state.MergeComplianceText;
        GatherDedupPolicy = string.IsNullOrWhiteSpace(state.DedupPolicy) ? "line" : state.DedupPolicy;

        foreach (var src in state.StagedSources.Where(x => !string.IsNullOrWhiteSpace(x)).Distinct(StringComparer.OrdinalIgnoreCase))
            GatherStagedSources.Add(src);

        foreach (var entry in state.SourceEntries.Where(x => !string.IsNullOrWhiteSpace(x.Path)))
        {
            var vm = new GatherSourceEntryViewModel(
                entry.Path,
                string.IsNullOrWhiteSpace(entry.Provider) ? "Unknown" : entry.Provider,
                string.IsNullOrWhiteSpace(entry.LicenseLabel) ? "unknown" : entry.LicenseLabel,
                entry.IsLicensePermitted);
            vm.IsEnabled = entry.IsEnabled;
            vm.Weight = entry.Weight < 1 ? 1 : entry.Weight;
            vm.PropertyChanged += (_, _) =>
            {
                OnPropertyChanged(nameof(CanGatherConvertParquet));
                OnPropertyChanged(nameof(GatherNeedsParquetConversion));
                OnPropertyChanged(nameof(GatherParquetHintText));
                OnPropertyChanged(nameof(CanGatherMergeSources));
                OnPropertyChanged(nameof(CanGatherValidateDataset));
            };
            GatherSourceEntries.Add(vm);
        }

        OnPropertyChanged(nameof(GatherSourcesCountText));
        OnPropertyChanged(nameof(CanGatherConvertParquet));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(CanGatherMergeSources));
        OnPropertyChanged(nameof(CanGatherValidateDataset));
    }

    private static T Clone<T>(T value)
    {
        var json = JsonSerializer.Serialize(value);
        return JsonSerializer.Deserialize<T>(json)!;
    }

    private string ResolveRunDirectoryForDebug()
        => string.IsNullOrWhiteSpace(RunDirectory)
            ? Path.Combine(Environment.CurrentDirectory, "runs", "default")
            : RunDirectory;

    private void ApplyCleaner(TextCleanerConfig config)
    {
        Cleaner.NormalizeNewLines = config.NormalizeNewLines;
        Cleaner.TrimLines = config.TrimLines;
        Cleaner.Lowercase = config.Lowercase;
        Cleaner.RemoveEmptyLines = config.RemoveEmptyLines;
    }

    private void ApplyTokenizer(TokenizerConfig config)
    {
        TokenizerConfig.Kind = config.Kind;
        TokenizerConfig.LowercaseBeforeTraining = config.LowercaseBeforeTraining;
        TokenizerConfig.TargetVocabSize = config.TargetVocabSize;
        TokenizerConfig.MinFrequency = config.MinFrequency;
        TokenizerConfig.MaxMerges = config.MaxMerges;
        TokenizerConfig.KeepPunctuationAsTokens = config.KeepPunctuationAsTokens;
        TokenizerConfig.UseCharacterFallback = config.UseCharacterFallback;
        TokenizerConfig.Notes = config.Notes;
        SelectedTokenizerKind = config.Kind;
    }

    private void ApplyModel(ModelConfig config)
    {
        ModelConfig.VocabSize = config.VocabSize;
        ModelConfig.BlockSize = config.BlockSize;
        ModelConfig.Layers = config.Layers;
        ModelConfig.Heads = config.Heads;
        ModelConfig.EmbeddingSize = config.EmbeddingSize;
        ModelConfig.Dropout = config.Dropout;
        RaiseModelTrainingBindingsChanged();
    }

    private void ApplyTraining(TrainingConfig config)
    {
        TrainingConfig.BatchSize = config.BatchSize;
        TrainingConfig.MaxSteps = config.MaxSteps;
        TrainingConfig.LearningRate = config.LearningRate;
        TrainingConfig.EvalEvery = config.EvalEvery;
        TrainingConfig.TrainSplit = config.TrainSplit;
        TrainingConfig.EnableGradientClipping = config.EnableGradientClipping;
        TrainingConfig.GradientClipNorm = config.GradientClipNorm;
        TrainingConfig.Optimizer = config.Optimizer;
        TrainingConfig.Scheduler = config.Scheduler;
        TrainingConfig.WarmupSteps = config.WarmupSteps;
        TrainingConfig.MixedPrecision = config.MixedPrecision;
        TrainingConfig.Precision = config.Precision;
        TrainingConfig.CheckpointEvery = config.CheckpointEvery;
        TrainingConfig.EnablePostTrainingQuantization = config.EnablePostTrainingQuantization;
        TrainingConfig.QuantizationProfile = config.QuantizationProfile;
        TrainingConfig.QuantizationCalibrationSamples = config.QuantizationCalibrationSamples;
        TrainingConfig.EnableQatPath = config.EnableQatPath;
        TrainingConfig.QatFineTuneSteps = Math.Max(1, config.QatFineTuneSteps);
        TrainingConfig.EnableDeduplication = config.EnableDeduplication;
        TrainingConfig.RemoveDuplicateLines = config.RemoveDuplicateLines;
        TrainingConfig.RemoveDuplicateParagraphs = config.RemoveDuplicateParagraphs;
        TrainingConfig.NormalizeUnicode = config.NormalizeUnicode;
        TrainingConfig.CollapseWhitespace = config.CollapseWhitespace;
        TrainingConfig.CurriculumLearning = config.CurriculumLearning;
        TrainingConfig.CurriculumWarmupRatio = config.CurriculumWarmupRatio;
        TrainingConfig.ResumeDatasetFromState = config.ResumeDatasetFromState;
        TrainingConfig.DeterministicShardShuffle = config.DeterministicShardShuffle;
        TrainingConfig.DataShuffleSeed = config.DataShuffleSeed;
        TrainingConfig.ClusterProfileName = config.ClusterProfileName;
        TrainingConfig.ClusterOrchestrator = config.ClusterOrchestrator;
        TrainingConfig.ClusterWorldSize = config.ClusterWorldSize;
        TrainingConfig.ClusterMaxRetries = config.ClusterMaxRetries;
        TrainingConfig.ClusterHeartbeatSeconds = config.ClusterHeartbeatSeconds;
        TrainingConfig.OrchestratePipelineStages = config.OrchestratePipelineStages;
        TrainingConfig.PipelineRunDataStage = config.PipelineRunDataStage;
        TrainingConfig.PipelineRunPreprocessStage = config.PipelineRunPreprocessStage;
        TrainingConfig.PipelineRunTrainStage = config.PipelineRunTrainStage;
        TrainingConfig.PipelineRunEvalStage = config.PipelineRunEvalStage;
        TrainingConfig.DistributedTraining = config.DistributedTraining;
        TrainingConfig.MultiGpuStrategy = string.IsNullOrWhiteSpace(config.MultiGpuStrategy) ? "none" : config.MultiGpuStrategy;
        TrainingConfig.GradientAccumulationSteps = Math.Max(1, config.GradientAccumulationSteps);
        TrainingConfig.AutoDeviceMap = config.AutoDeviceMap;
        TrainingConfig.AlignmentMode = config.AlignmentMode;
        TrainingConfig.FineTuningOrchestration = config.FineTuningOrchestration;
        TrainingConfig.FineTuneStageSft = config.FineTuneStageSft;
        TrainingConfig.FineTuneStageDpo = config.FineTuneStageDpo;
        TrainingConfig.FineTuneStageRlhf = config.FineTuneStageRlhf;
        TrainingConfig.RlhfFeedbackSource = string.IsNullOrWhiteSpace(config.RlhfFeedbackSource) ? "inline" : config.RlhfFeedbackSource;
        TrainingConfig.RlhfFeedbackPath = config.RlhfFeedbackPath ?? string.Empty;
        TrainingConfig.RewardModelingEnabled = config.RewardModelingEnabled;
        TrainingConfig.SafetyPolicyMode = string.IsNullOrWhiteSpace(config.SafetyPolicyMode) ? "standard" : config.SafetyPolicyMode;
        TrainingConfig.ExportOnnx = config.ExportOnnx;
        TrainingConfig.ExportGguf = config.ExportGguf;
        TrainingConfig.EvalSuite = config.EvalSuite;
        TrainingConfig.ForceCpu = config.ForceCpu;
        RaiseModelTrainingBindingsChanged();
    }

    private string InferTrainingProfileFromConfig(TrainingConfig cfg)
    {
        if (cfg is null) return "Custom";

        if (cfg.DistributedTraining || string.Equals(cfg.MultiGpuStrategy, "ddp", StringComparison.OrdinalIgnoreCase) || string.Equals(cfg.MultiGpuStrategy, "fsdp", StringComparison.OrdinalIgnoreCase))
            return "Cluster";

        if (cfg.MaxSteps >= 1500
            && string.Equals(cfg.Optimizer, "adafactor", StringComparison.OrdinalIgnoreCase)
            && string.Equals(cfg.Precision, "bf16", StringComparison.OrdinalIgnoreCase)
            && cfg.FineTuningOrchestration
            && cfg.FineTuneStageRlhf)
            return "Research";

        if (cfg.MaxSteps >= 1000
            && string.Equals(cfg.Optimizer, "adamw", StringComparison.OrdinalIgnoreCase)
            && string.Equals(cfg.Scheduler, "cosine", StringComparison.OrdinalIgnoreCase)
            && string.Equals(cfg.Precision, "fp16", StringComparison.OrdinalIgnoreCase))
            return "Serious";

        if (cfg.MaxSteps <= 250 && string.Equals(cfg.Scheduler, "none", StringComparison.OrdinalIgnoreCase))
            return "Tiny";

        if (cfg.MaxSteps >= 400
            && string.Equals(cfg.Optimizer, "adamw", StringComparison.OrdinalIgnoreCase)
            && string.Equals(cfg.Scheduler, "cosine", StringComparison.OrdinalIgnoreCase))
            return "Balanced";

        return "Custom";
    }

    private void NormalizeLoadedTrainingProfileConsistency()
    {
        var profile = (SelectedTrainingProfile ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(profile) || profile.Equals("Custom", StringComparison.OrdinalIgnoreCase))
            return;

        var isConsistent = profile switch
        {
            "Serious" => string.Equals(TrainingConfig.Optimizer, "adamw", StringComparison.OrdinalIgnoreCase)
                         && string.Equals(TrainingConfig.Scheduler, "cosine", StringComparison.OrdinalIgnoreCase)
                         && string.Equals(TrainingConfig.Precision, "fp16", StringComparison.OrdinalIgnoreCase)
                         && TrainingConfig.MaxSteps >= 3000,
            "Balanced" => string.Equals(TrainingConfig.Optimizer, "adamw", StringComparison.OrdinalIgnoreCase)
                          && string.Equals(TrainingConfig.Scheduler, "cosine", StringComparison.OrdinalIgnoreCase),
            "Tiny" => TrainingConfig.MaxSteps <= 300 && string.Equals(TrainingConfig.Scheduler, "none", StringComparison.OrdinalIgnoreCase),
            "Research" => string.Equals(TrainingConfig.Optimizer, "adafactor", StringComparison.OrdinalIgnoreCase)
                          && TrainingConfig.FineTuneStageRlhf
                          && TrainingConfig.FineTuningOrchestration,
            "Cluster" => TrainingConfig.DistributedTraining || !string.Equals(TrainingConfig.MultiGpuStrategy, "none", StringComparison.OrdinalIgnoreCase),
            _ => true
        };

        if (isConsistent) return;

        // Avoid misleading UI: if values do not match the shown preset, show Custom.
        _selectedTrainingProfile = "Custom";
        OnPropertyChanged(nameof(SelectedTrainingProfile));
        OnPropertyChanged(nameof(TrainingProfileHintText));
    }

    private void ApplySampling(SamplingConfig config)
    {
        SamplingConfig.Temperature = config.Temperature;
        SamplingConfig.TopK = config.TopK;
        SamplingConfig.Seed = config.Seed;
        SamplingConfig.Greedy = config.Greedy;
        SamplingConfig.MaxNewTokens = config.MaxNewTokens;
        OnPropertyChanged(nameof(GenerationOutputWindowText));
    }

    private string BuildChartSummary()
    {
        if (TrainingLogs.Count == 0) return "No chart data yet.";
        var last = TrainingLogs.Last();
        var avgTok = TrainingLogs.Average(x => x.TokensPerSecond);
        return $"Last step={last.Step} | train={last.TrainLoss:F4} | val={last.ValLoss:F4} | tok/s(avg)={avgTok:F1}";
    }

    private void ApplyRecommendedTokenizerSettings(TokenizerKind kind)
    {
        GuidedDefaultsEngine.ApplyTokenizerPreset(kind, DatasetText, TokenizerConfig);
    }

    private void ApplyRecommendedModelAndTrainingSettings(TokenizerKind kind)
    {
        GuidedDefaultsEngine.ApplyModelTrainingPresetForTokenizer(kind, ModelConfig, TrainingConfig);
        RaiseModelTrainingBindingsChanged();
    }

    private void ApplyTrainingProfile(string profile)
    {
        GuidedDefaultsEngine.ApplyTrainingProfile(profile, TrainingConfig);
    }

    private void RaiseModelTrainingBindingsChanged()
    {
        OnPropertyChanged(nameof(ModelVocabSize));
        OnPropertyChanged(nameof(ModelBlockSize));
        OnPropertyChanged(nameof(ModelContextWindowText));
        OnPropertyChanged(nameof(ModelLayers));
        OnPropertyChanged(nameof(ModelHeads));
        OnPropertyChanged(nameof(ModelEmbeddingSize));
        OnPropertyChanged(nameof(TrainingBatchSize));
        OnPropertyChanged(nameof(TrainingMaxSteps));
        OnPropertyChanged(nameof(TrainingLearningRate));
        OnPropertyChanged(nameof(TrainingEvalEvery));
        OnPropertyChanged(nameof(TrainingForceCpu));
        OnPropertyChanged(nameof(TrainingOptimizer));
        OnPropertyChanged(nameof(TrainingScheduler));
        OnPropertyChanged(nameof(TrainingWarmupSteps));
        OnPropertyChanged(nameof(TrainingMixedPrecision));
        OnPropertyChanged(nameof(TrainingPrecision));
        OnPropertyChanged(nameof(TrainingGradientClipping));
        OnPropertyChanged(nameof(TrainingGradientClipNorm));
        OnPropertyChanged(nameof(TrainingCheckpointEvery));
        OnPropertyChanged(nameof(TrainingEnablePostQuantization));
        OnPropertyChanged(nameof(TrainingQuantizationProfile));
        OnPropertyChanged(nameof(TrainingQuantizationCalibrationSamples));
        OnPropertyChanged(nameof(TrainingEnableQatPath));
        OnPropertyChanged(nameof(TrainingQatFineTuneSteps));
        OnPropertyChanged(nameof(TrainingDedup));
        OnPropertyChanged(nameof(TrainingDedupLines));
        OnPropertyChanged(nameof(TrainingDedupParagraphs));
        OnPropertyChanged(nameof(TrainingNormalizeUnicode));
        OnPropertyChanged(nameof(TrainingCollapseWhitespace));
        OnPropertyChanged(nameof(TrainingCurriculumLearning));
        OnPropertyChanged(nameof(TrainingCurriculumWarmupRatio));
        OnPropertyChanged(nameof(TrainingOrchestratePipelineStages));
        OnPropertyChanged(nameof(TrainingPipelineRunDataStage));
        OnPropertyChanged(nameof(TrainingPipelineRunPreprocessStage));
        OnPropertyChanged(nameof(TrainingPipelineRunTrainStage));
        OnPropertyChanged(nameof(TrainingPipelineRunEvalStage));
        OnPropertyChanged(nameof(TrainingClusterProfileName));
        OnPropertyChanged(nameof(TrainingDistributed));
        OnPropertyChanged(nameof(TrainingMultiGpuStrategy));
        OnPropertyChanged(nameof(TrainingGradientAccumulationSteps));
        OnPropertyChanged(nameof(TrainingAutoDeviceMap));
        OnPropertyChanged(nameof(TrainingAlignmentMode));
        OnPropertyChanged(nameof(TrainingFineTuningOrchestration));
        OnPropertyChanged(nameof(TrainingFineTuneStageSft));
        OnPropertyChanged(nameof(TrainingFineTuneStageDpo));
        OnPropertyChanged(nameof(TrainingFineTuneStageRlhf));
        OnPropertyChanged(nameof(TrainingRlhfFeedbackSource));
        OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
        OnPropertyChanged(nameof(TrainingRewardModelingEnabled));
        OnPropertyChanged(nameof(TrainingSafetyPolicyMode));
        OnPropertyChanged(nameof(TrainingExportOnnx));
        OnPropertyChanged(nameof(TrainingExportGguf));
        OnPropertyChanged(nameof(TrainingEvalSuite));
        OnPropertyChanged(nameof(TrainingWizardDetailedText));
        OnPropertyChanged(nameof(TrainingPreflightText));
        OnPropertyChanged(nameof(ParameterEstimate));
    }

    private string BuildTokenizerRecommendationText()
    {
        return SelectedTokenizerKind switch
        {
            TokenizerKind.Character => "Preset Character: vocab 256, min freq 1, merges 0. Utile per dataset piccolo o testi rumorosi.",
            TokenizerKind.Word => "Preset Word: vocab proporzionale al dataset, min freq 2, merges 0. Più leggibile ma più OOV su parole nuove.",
            TokenizerKind.ByteLevelBpe => "Preset Byte-level BPE: robusto su UTF-8 e simboli, con merges medi/alti per corpus reali.",
            TokenizerKind.Unigram => "Preset Unigram: subword probabilistico, efficace su corpora vari e testi misti.",
            TokenizerKind.WordPiece => "Preset WordPiece: subword con prefisso ##, utile per pipeline di fine-tuning NLP.",
            TokenizerKind.SimpleBpe => "Preset BPE: vocab e merges medi, min freq 2. Buon compromesso tra robustezza e compressione.",
            TokenizerKind.HybridFallback => "Preset Hybrid: come BPE ma con fallback char per OOV. Più robusto in generazione reale.",
            TokenizerKind.HierarchicalExperimental => "Preset Experimental: parametri alti per esplorazione macro/subword; non consigliato per produzione.",
            _ => string.Empty
        };
    }

    private string BuildTokenizerPresetBadgeText()
    {
        return SelectedTokenizerKind switch
        {
            TokenizerKind.Character => "Mode: Educational",
            TokenizerKind.Word => "Mode: Basic NLP",
            TokenizerKind.SimpleBpe => "Mode: Balanced",
            TokenizerKind.ByteLevelBpe => "Mode: Robust UTF-8",
            TokenizerKind.Unigram => "Mode: Probabilistic Subword",
            TokenizerKind.WordPiece => "Mode: Fine-tuning Ready",
            TokenizerKind.HybridFallback => "Mode: Production-safe Local",
            TokenizerKind.HierarchicalExperimental => "Mode: Experimental",
            _ => "Mode: Custom"
        };
    }

    private string BuildTokenizerIdealValuesText()
    {
        var probe = Clone(TokenizerConfig);
        GuidedDefaultsEngine.ApplyTokenizerPreset(SelectedTokenizerKind, DatasetText, probe);
        var fallback = probe.UseCharacterFallback ? "on" : "off";
        return $"Ideal values: vocab {probe.TargetVocabSize:N0}, min freq {probe.MinFrequency}, merges {probe.MaxMerges:N0}, char fallback {fallback}.";
    }

    private string BuildRoadmapChecklistText()
    {
        return string.Join('\n',
            "Roadmap Checklist (in-app):",
            "- Advanced Training panel: completed/verified",
            "- Guided defaults engine: completed/verified",
            "- Advanced tokenizer options: byte-level BPE + unigram + wordpiece",
            "- Multi-GPU/cluster/RLHF: planned next blocks");
    }

    private string BuildTrainingPreflightText()
    {
        var notes = new List<string>();

        if (TrainingConfig.DistributedTraining)
        {
            var wsRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
            if (!int.TryParse(wsRaw, out var ws) || ws <= 1)
                notes.Add("Distributed ON: set WORLD_SIZE>1 before start.");
            else
                notes.Add($"Distributed ON: WORLD_SIZE={ws}.");
            notes.Add($"Multi-GPU strategy: {TrainingConfig.MultiGpuStrategy}.");
        }
        else
        {
            notes.Add("Distributed OFF: single-device mode.");
        }
        if (TrainingConfig.OrchestratePipelineStages)
        {
            notes.Add($"Pipeline ON: data={TrainingConfig.PipelineRunDataStage}, preprocess={TrainingConfig.PipelineRunPreprocessStage}, train={TrainingConfig.PipelineRunTrainStage}, eval={TrainingConfig.PipelineRunEvalStage}.");
        }
        else
        {
            notes.Add("Pipeline OFF: single train stage.");
        }

        notes.Add($"Grad accumulation: x{Math.Max(1, TrainingConfig.GradientAccumulationSteps)}.");
        notes.Add($"Auto device map: {(TrainingConfig.AutoDeviceMap ? "ON" : "OFF")}.");
        if (TrainingConfig.FineTuningOrchestration)
            notes.Add($"Fine-tuning pipeline ON: sft={TrainingConfig.FineTuneStageSft}, dpo={TrainingConfig.FineTuneStageDpo}, rlhf={TrainingConfig.FineTuneStageRlhf}.");
        else
            notes.Add($"Fine-tuning pipeline OFF: alignment mode={TrainingConfig.AlignmentMode}.");
        if (TrainingConfig.FineTuneStageRlhf && string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase))
            notes.Add($"RLHF inline feedback collected: {RlhfCollectedFeedback.Count}.");
        notes.Add($"Reward model: {(TrainingConfig.RewardModelingEnabled ? "ON" : "OFF")}.");
        notes.Add($"Safety policy: {TrainingConfig.SafetyPolicyMode}.");
        notes.Add($"Export targets: onnx={TrainingConfig.ExportOnnx}, gguf={TrainingConfig.ExportGguf}.");
        if (TrainingConfig.ExportGguf)
            notes.Add("Ollama auto-package: ON (exports/ollama after training if GGUF exists).");
        notes.Add($"Cluster profile: {TrainingConfig.ClusterProfileName} ({TrainingConfig.ClusterOrchestrator}, world={TrainingConfig.ClusterWorldSize}).");

        notes.Add($"Eval pack: {TrainingConfig.EvalSuite}");

        if (TrainingConfig.EnablePostTrainingQuantization && !TrainingConfig.ForceCpu)
            notes.Add("Quantization ON but Force CPU OFF (artifact may be skipped).");
        if (TrainingConfig.EnablePostTrainingQuantization)
            notes.Add($"Quant profile: {TrainingConfig.QuantizationProfile}, calib={TrainingConfig.QuantizationCalibrationSamples}.");
        if (TrainingConfig.EnableQatPath)
            notes.Add($"QAT path ON: fine-tune steps={TrainingConfig.QatFineTuneSteps}.");

        if (TrainingConfig.MixedPrecision && TrainingConfig.ForceCpu)
            notes.Add("Mixed precision ON with Force CPU (no CUDA AMP gain).");

        return string.Join(" ", notes);
    }

    private string BuildTrainingWizardDetailedText()
    {
        var lines = new List<string>
        {
            "Guided Wizards:",
            $"1) Multi-GPU setup: {(TrainingConfig.DistributedTraining && !string.Equals(TrainingConfig.MultiGpuStrategy, "none", StringComparison.OrdinalIgnoreCase) ? "OK" : "Configure Distributed + strategy")}.",
            $"2) Cluster profile: {(string.IsNullOrWhiteSpace(TrainingConfig.ClusterProfileName) ? "Select profile" : $"OK ({TrainingConfig.ClusterProfileName})")}.",
            $"3) RLHF import: {(TrainingConfig.FineTuneStageRlhf ? (string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath) && !string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase) ? "Set feedback path" : "OK") : "Optional")}.",
            $"4) Eval pack: {(string.IsNullOrWhiteSpace(TrainingConfig.EvalSuite) ? "Select eval suite" : $"OK ({TrainingConfig.EvalSuite})")}."
        };
        return string.Join("\n", lines);
    }

    private string? GetBlockingPreflightIssue()
    {
        if (TrainingConfig.FineTuningOrchestration
            && !TrainingConfig.FineTuneStageSft
            && !TrainingConfig.FineTuneStageDpo
            && !TrainingConfig.FineTuneStageRlhf)
            return "Fine-tuning orchestration is ON but no SFT/DPO/RLHF stage is selected.";
        if (TrainingConfig.FineTuneStageRlhf
            && !string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)
            && string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath))
            return "RLHF stage requires feedback path when feedback source is not inline.";
        if (TrainingConfig.FineTuneStageRlhf
            && string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)
            && RlhfCollectedFeedback.Count == 0
            && string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath))
            return "RLHF inline mode requires collected feedback records. Add at least one record in Advanced Training.";

        if (TrainingConfig.OrchestratePipelineStages && !TrainingConfig.PipelineRunTrainStage)
            return "Pipeline orchestration is ON but Train stage is disabled. Enable train stage or disable orchestration.";

        if (!TrainingConfig.DistributedTraining) return null;

        var wsRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
        if (!int.TryParse(wsRaw, out var ws) || ws <= 1)
            return "Distributed training requires WORLD_SIZE > 1. Set WORLD_SIZE/RANK/LOCAL_RANK and retry, or disable Distributed mode.";
        if (TrainingConfig.ClusterWorldSize > 1 && ws != TrainingConfig.ClusterWorldSize)
            return $"Distributed preflight mismatch: WORLD_SIZE={ws} but ClusterWorldSize={TrainingConfig.ClusterWorldSize}. Align values before training.";
        if (string.Equals(TrainingConfig.MultiGpuStrategy, "none", StringComparison.OrdinalIgnoreCase))
            return "Distributed mode is ON but Multi-GPU strategy is 'none'. Select 'ddp' or 'fsdp', or disable Distributed mode.";

        return null;
    }

    private static string ResolveProjectRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var sln = Path.Combine(dir.FullName, "LLMForgeStudio.sln");
            var backendDir = Path.Combine(dir.FullName, "backends", "python");
            if (File.Exists(sln) && Directory.Exists(backendDir))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private void GoToWizardTarget()
    {
        if (!_wizardSetupDone) { SelectedSection = "Hardware"; return; }
        if (!_wizardDatasetImported) { SelectedSection = "Dataset"; return; }
        if (!_wizardTokenizerTrained || !_wizardPreviewBuilt) { SelectedSection = "Tokenization"; return; }
        if (!_wizardTrainingStarted) { SelectedSection = "Training"; return; }
        SelectedSection = IsEnglish ? "Generation" : "Generazione";
    }

    private void PrepareOllamaFineTuneRun()
    {
        var baseModelPath = ResolveDefaultOllamaBaseModelPath();
        OllamaFtBaseModelPath = baseModelPath;
        if (string.IsNullOrWhiteSpace(OllamaFtDatasetPath) && !string.IsNullOrWhiteSpace(DatasetPath))
        {
            OllamaFtDatasetPath = DatasetPath;
        }
        var outDir = string.IsNullOrWhiteSpace(OllamaFtOutputDirectory)
            ? Path.Combine(RunDirectory, "fine_tuning_ollama")
            : OllamaFtOutputDirectory;
        OllamaFtOutputDirectory = outDir;

        var lines = new List<string>
        {
            "Fine-Tuning (Ollama) workspace prepared.",
            $"Base model path: {(string.IsNullOrWhiteSpace(OllamaFtBaseModelPath) ? "(missing)" : OllamaFtBaseModelPath)}",
            $"New model name: {OllamaFtOutputModelName}",
            $"Dataset path: {(string.IsNullOrWhiteSpace(OllamaFtDatasetPath) ? "(missing)" : OllamaFtDatasetPath)}",
            $"Output directory: {OllamaFtOutputDirectory}",
            $"Backend: {OllamaFtBackend}",
            $"Method: {OllamaFtMethod}, template: {OllamaFtTemplate}",
            $"Epochs: {OllamaFtEpochs}, batch: {OllamaFtBatchSize}, grad accum: {OllamaFtGradientAccumulation}",
            $"LR: {OllamaFtLearningRate}, LoRA rank/alpha/dropout: {OllamaFtLoraRank}/{OllamaFtLoraAlpha}/{OllamaFtLoraDropout}",
            $"Pack for Ollama: {OllamaFtPackForOllama}"
        };
        OllamaFtStatusText = "Workspace prepared.";
        Log = string.Join('\n', lines);
        OnPropertyChanged(nameof(CanConvertOllamaFineTuneToGguf));
        OnPropertyChanged(nameof(CanFinalizeOllamaFineTuneExport));
        RaiseOllamaFtPipelineChanged();
    }

    private async Task StartOllamaFineTuneAsync()
    {
        if (IsOllamaFtRunning) return;
        if (!PythonBackendBridge.IsPythonAvailable(PythonPath))
        {
            OllamaFtStatusText = "Python backend not available.";
            Log = $"Python non trovato: {PythonPath}";
            return;
        }

        PrepareOllamaFineTuneRun();
        if (string.IsNullOrWhiteSpace(OllamaFtDatasetPath))
        {
            OllamaFtStatusText = "Dataset required.";
            Log = "Fine-tuning (Ollama) blocked: dataset path is missing.";
            return;
        }
        if (string.IsNullOrWhiteSpace(OllamaFtBaseModelPath) || !File.Exists(OllamaFtBaseModelPath))
        {
            OllamaFtStatusText = "Base model required.";
            Log = "Fine-tuning (Ollama) blocked: default base model checkpoint was not found in run directory (model.pt).";
            return;
        }
        if (string.IsNullOrWhiteSpace(OllamaFtOutputModelName))
        {
            OllamaFtStatusText = "Output model name required.";
            Log = "Fine-tuning (Ollama) blocked: set a new output model name.";
            return;
        }

        Directory.CreateDirectory(OllamaFtOutputDirectory);
        var jobPath = Path.Combine(OllamaFtOutputDirectory, "ollama_finetune_job.json");
        var job = new
        {
            base_model_path = OllamaFtBaseModelPath,
            output_model_name = OllamaFtOutputModelName.Trim(),
            dataset_path = OllamaFtDatasetPath,
            output_directory = OllamaFtOutputDirectory,
            backend = OllamaFtBackend,
            method = OllamaFtMethod,
            template = OllamaFtTemplate,
            epochs = OllamaFtEpochs,
            batch_size = OllamaFtBatchSize,
            grad_accum = OllamaFtGradientAccumulation,
            learning_rate = OllamaFtLearningRate,
            lora_rank = OllamaFtLoraRank,
            lora_alpha = OllamaFtLoraAlpha,
            lora_dropout = OllamaFtLoraDropout,
            pack_for_ollama = OllamaFtPackForOllama
        };
        await File.WriteAllTextAsync(jobPath, JsonSerializer.Serialize(job, new JsonSerializerOptions { WriteIndented = true }));

        var projectRoot = ResolveProjectRoot();
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "ollama_finetune_runner.py");
        var startInfo = PythonBackendBridge.CreateStartInfo(PythonPath, scriptPath, $"--job \"{jobPath}\"");

        _ollamaFtProcess = Process.Start(startInfo);
        if (_ollamaFtProcess is null)
        {
            OllamaFtStatusText = "Fine-tune process failed to start.";
            return;
        }

        IsOllamaFtRunning = true;
        OllamaFtStatusText = "Fine-tuning started...";
        RaiseOllamaFtPipelineChanged();
        var stdOutTask = _ollamaFtProcess.StandardOutput.ReadToEndAsync();
        var stdErrTask = _ollamaFtProcess.StandardError.ReadToEndAsync();
        _ = MonitorOllamaFineTuneAsync(_ollamaFtProcess, stdOutTask, stdErrTask);
    }

    private async Task MonitorOllamaFineTuneAsync(Process process, Task<string> stdOutTask, Task<string> stdErrTask)
    {
        try
        {
            await process.WaitForExitAsync();
            var stdOut = await stdOutTask;
            var stdErr = await stdErrTask;
            if (process.ExitCode == 0)
            {
                OllamaFtStatusText = "Fine-tuning completed.";
                Log = $"Fine-tuning (Ollama) completed.\nOutput: {OllamaFtOutputDirectory}\n{stdOut}".Trim();
                if (OllamaFtPackForOllama)
                {
                    await ConvertOllamaFineTuneToGgufAsync();
                    if (CanFinalizeOllamaFineTuneExport)
                        FinalizeOllamaFineTuneExport();
                }
            }
            else
            {
                var err = string.IsNullOrWhiteSpace(stdErr) ? stdOut : stdErr;
                OllamaFtStatusText = $"Fine-tuning failed (exit {process.ExitCode}).";
                Log = $"Fine-tuning (Ollama) failed.\n{err}".Trim();
            }
        }
        finally
        {
            IsOllamaFtRunning = false;
            _ollamaFtProcess = null;
            OnPropertyChanged(nameof(CanConvertOllamaFineTuneToGguf));
            OnPropertyChanged(nameof(CanFinalizeOllamaFineTuneExport));
            RaiseOllamaFtPipelineChanged();
        }
    }

    private void CancelOllamaFineTune()
    {
        try
        {
            if (_ollamaFtProcess is not null && !_ollamaFtProcess.HasExited)
                _ollamaFtProcess.Kill(true);
            OllamaFtStatusText = "Fine-tuning cancelled.";
        }
        catch (Exception ex)
        {
            OllamaFtStatusText = $"Cancel failed: {ex.Message}";
        }
    }

    private async Task ConvertOllamaFineTuneToGgufAsync()
    {
        var manifestPath = Path.Combine(OllamaFtOutputDirectory, "ollama_finetune_manifest.json");
        if (!File.Exists(manifestPath))
        {
            OllamaFtStatusText = "Run fine-tuning first.";
            return;
        }

        var exportsDir = Path.Combine(OllamaFtOutputDirectory, "exports", "ollama_finetune");
        Directory.CreateDirectory(exportsDir);
        var statusPath = Path.Combine(exportsDir, "ollama_handoff_status.json");
        var ggufPath = Path.Combine(exportsDir, "model.gguf");
        var converterStatusPath = Path.Combine(exportsDir, "gguf_converter_status.json");
        var stageStatePath = Path.Combine(exportsDir, "ollama_pipeline_stage_state.json");
        var runtimeSnapshotPath = Path.Combine(exportsDir, "runtime_environment_snapshot.json");
        var runtimeLockPath = Path.Combine(exportsDir, "runtime_lock_profile.json");
        var runtimeMismatchPath = Path.Combine(exportsDir, "runtime_preflight_mismatch.json");

        // Idempotent resume: if GGUF already exists, mark conversion as completed and skip.
        if (File.Exists(ggufPath) && new FileInfo(ggufPath).Length > 0)
        {
            WritePipelineStageState(stageStatePath, "convert_gguf", "completed", "Existing GGUF detected. Conversion step skipped.");
            OllamaFtStatusText = "GGUF conversion completed (reused existing artifact).";
            Log = "Conversion step skipped: existing model.gguf found.";
            OnPropertyChanged(nameof(CanFinalizeOllamaFineTuneExport));
            RaiseOllamaFtPipelineChanged();
            return;
        }

        WritePipelineStageState(stageStatePath, "convert_gguf", "running", "Starting GGUF conversion...");
        var projectRoot = ResolveProjectRoot();
        var converterEnv = Environment.GetEnvironmentVariable("LLMFORGE_GGUF_CONVERTER") ?? string.Empty;
        var bundledCandidates = new[]
        {
            Path.Combine(projectRoot, "backends", "python", "tools", "gguf_converter.exe"),
            Path.Combine(projectRoot, "backends", "python", "tools", "gguf_converter.bat"),
            Path.Combine(projectRoot, "backends", "python", "tools", "gguf_converter.cmd"),
            Path.Combine(projectRoot, "backends", "python", "tools", "gguf_converter.sh")
        };
        var bundledResolved = bundledCandidates.FirstOrDefault(File.Exists) ?? string.Empty;
        var currentProfile = new RuntimeLockProfile(
            pythonPath: PythonPath,
            converterEnv: converterEnv,
            converterBundledResolved: bundledResolved,
            backend: OllamaFtBackend,
            method: OllamaFtMethod,
            template: OllamaFtTemplate);

        if (File.Exists(runtimeLockPath))
        {
            var (hasMismatch, mismatchDetails) = TryDetectRuntimeMismatch(runtimeLockPath, currentProfile);
            if (hasMismatch)
            {
                var mismatch = new
                {
                    generatedAtUtc = DateTimeOffset.UtcNow,
                    status = "blocked",
                    reason = "runtime-lock-mismatch",
                    message = "Runtime mismatch detected. Keep Python/converter/backend settings consistent or regenerate a fresh run lock.",
                    details = mismatchDetails
                };
                await File.WriteAllTextAsync(runtimeMismatchPath, JsonSerializer.Serialize(mismatch, new JsonSerializerOptions { WriteIndented = true }));
                WritePipelineStageState(stageStatePath, "convert_gguf", "blocked", "Runtime lock mismatch.");
                OllamaFtStatusText = "GGUF conversion blocked [RUNTIME_LOCK_MISMATCH]";
                Log = $"GGUF conversion blocked by runtime mismatch.\nDetails: {string.Join("; ", mismatchDetails)}\nSee: {runtimeMismatchPath}";
                return;
            }
        }
        else
        {
            await File.WriteAllTextAsync(runtimeLockPath, JsonSerializer.Serialize(currentProfile, new JsonSerializerOptions { WriteIndented = true }));
        }

        var runtimeSnapshot = new
        {
            generatedAtUtc = DateTimeOffset.UtcNow,
            pythonPath = PythonPath,
            converterEnv,
            converterBundledResolved = bundledResolved,
            converterResolutionOrder = new[] { "env", "bundled", "fallback-copy-existing-gguf" },
            runtimeLockPath,
            outputDirectory = OllamaFtOutputDirectory,
            exportsDirectory = exportsDir
        };
        await File.WriteAllTextAsync(runtimeSnapshotPath, JsonSerializer.Serialize(runtimeSnapshot, new JsonSerializerOptions { WriteIndented = true }));
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "gguf_converter_runtime.py");
        var startInfo = PythonBackendBridge.CreateStartInfo(PythonPath, scriptPath,
            $"--input-dir \"{OllamaFtOutputDirectory}\" --output \"{ggufPath}\" --status \"{converterStatusPath}\" --attempts 2");
        using var process = Process.Start(startInfo);
        if (process is null)
        {
            WritePipelineStageState(stageStatePath, "convert_gguf", "failed", "Process start failure.");
            OllamaFtStatusText = "GGUF conversion failed to start.";
            return;
        }
        var outText = await process.StandardOutput.ReadToEndAsync();
        var errText = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        if (process.ExitCode != 0 || !File.Exists(ggufPath))
        {
            var (code, message) = ReadGgufConverterFailure(converterStatusPath);
            WritePipelineStageState(stageStatePath, "convert_gguf", "failed", $"Converter failed [{code}] {message}");
            OllamaFtStatusText = $"GGUF conversion failed [{code}]";
            Log = $"GGUF conversion failed [{code}]. {message}\n{(string.IsNullOrWhiteSpace(errText) ? outText : errText)}".Trim();
            return;
        }

        WritePipelineStageState(stageStatePath, "convert_gguf", "completed", "GGUF conversion completed.");
        OllamaFtStatusText = "GGUF conversion completed.";
        Log = "Conversion step completed. model.gguf generated.";
        OnPropertyChanged(nameof(CanFinalizeOllamaFineTuneExport));
        RaiseOllamaFtPipelineChanged();
    }

    private void FinalizeOllamaFineTuneExport()
    {
        var exportsDir = Path.Combine(OllamaFtOutputDirectory, "exports", "ollama_finetune");
        var ggufPath = Path.Combine(exportsDir, "model.gguf");
        var stageStatePath = Path.Combine(exportsDir, "ollama_pipeline_stage_state.json");
        var statusPath = Path.Combine(exportsDir, "ollama_handoff_status.json");
        var existingReady = TryReadHandoffReady(statusPath);
        if (existingReady)
        {
            WritePipelineStageState(stageStatePath, "finalize_export", "completed", "Already finalized (ready status found).");
            OllamaFtStatusText = "Final export already ready.";
            Log = $"Finalize step skipped: ready bundle already exists in {exportsDir}";
            RaiseOllamaFtPipelineChanged();
            return;
        }

        WritePipelineStageState(stageStatePath, "finalize_export", "running", "Starting finalization...");
        if (!File.Exists(ggufPath))
        {
            WritePipelineStageState(stageStatePath, "finalize_export", "blocked", "model.gguf missing.");
            OllamaFtStatusText = "Finalize blocked: model.gguf missing.";
            return;
        }

        var modelName = string.IsNullOrWhiteSpace(OllamaFtOutputModelName) ? "llmforge-finetuned" : OllamaFtOutputModelName.Trim();
        var modelfilePath = Path.Combine(exportsDir, "Modelfile");
        File.WriteAllText(modelfilePath, $"FROM ./model.gguf\nTEMPLATE \"{{{{ .Prompt }}}}\"\nPARAMETER temperature 0.7\n", Encoding.UTF8);

        var validationPath = Path.Combine(exportsDir, "ollama_bundle_validation.json");
        var validation = new
        {
            generatedAtUtc = DateTimeOffset.UtcNow,
            modelName,
            checks = new
            {
                modelGgufExists = File.Exists(ggufPath),
                modelfileExists = File.Exists(modelfilePath),
                ggufSizeBytes = new FileInfo(ggufPath).Length,
                modelfileSizeBytes = new FileInfo(modelfilePath).Length
            }
        };
        File.WriteAllText(validationPath, JsonSerializer.Serialize(validation, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);

        var bundleReady =
            File.Exists(ggufPath)
            && File.Exists(modelfilePath)
            && new FileInfo(ggufPath).Length > 0
            && new FileInfo(modelfilePath).Length > 0;

        var status = new
        {
            status = bundleReady ? "ready" : "blocked",
            modelName,
            message = bundleReady
                ? "Bundle ready for manual Ollama handoff."
                : "Finalize blocked: bundle validation failed.",
            generatedAtUtc = DateTimeOffset.UtcNow
        };
        File.WriteAllText(statusPath, JsonSerializer.Serialize(status, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
        WritePipelineStageState(stageStatePath, "finalize_export", bundleReady ? "completed" : "blocked",
            bundleReady ? "Final export ready." : "Bundle validation failed.");
        OllamaFtStatusText = bundleReady ? "Final export ready." : "Finalize blocked by validation.";
        Log = bundleReady
            ? $"Finalize step completed. Bundle ready in: {exportsDir}"
            : $"Finalize blocked. Check validation file: {validationPath}";
        RaiseOllamaFtPipelineChanged();
    }

    private static (string code, string message) ReadGgufConverterFailure(string statusPath)
    {
        try
        {
            if (!File.Exists(statusPath))
                return ("GGUF_STATUS_MISSING", "Converter status file missing.");
            using var doc = JsonDocument.Parse(File.ReadAllText(statusPath, Encoding.UTF8));
            var code = doc.RootElement.TryGetProperty("errorCode", out var c) ? c.GetString() ?? "GGUF_UNKNOWN" : "GGUF_UNKNOWN";
            var message = doc.RootElement.TryGetProperty("message", out var m) ? m.GetString() ?? "Unknown converter error." : "Unknown converter error.";
            return (code, message);
        }
        catch
        {
            return ("GGUF_STATUS_PARSE_ERROR", "Converter status parse failed.");
        }
    }

    private sealed record RuntimeLockProfile(
        string pythonPath,
        string converterEnv,
        string converterBundledResolved,
        string backend,
        string method,
        string template);

    private static void WritePipelineStageState(string path, string stage, string status, string message)
    {
        try
        {
            var payload = new
            {
                updatedAtUtc = DateTimeOffset.UtcNow,
                stage,
                status,
                message
            };
            File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }), Encoding.UTF8);
        }
        catch
        {
            // best effort
        }
    }

    private static bool TryReadHandoffReady(string statusPath)
    {
        try
        {
            if (!File.Exists(statusPath)) return false;
            using var doc = JsonDocument.Parse(File.ReadAllText(statusPath, Encoding.UTF8));
            if (!doc.RootElement.TryGetProperty("status", out var s)) return false;
            return string.Equals(s.GetString(), "ready", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }

    private static (bool hasMismatch, List<string> details) TryDetectRuntimeMismatch(string lockPath, RuntimeLockProfile current)
    {
        var details = new List<string>();
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(lockPath, Encoding.UTF8));
            var root = doc.RootElement;
            var lockedPython = root.TryGetProperty("pythonPath", out var p) ? p.GetString() ?? string.Empty : string.Empty;
            var lockedConverterEnv = root.TryGetProperty("converterEnv", out var e) ? e.GetString() ?? string.Empty : string.Empty;
            var lockedBundled = root.TryGetProperty("converterBundledResolved", out var b) ? b.GetString() ?? string.Empty : string.Empty;
            var lockedBackend = root.TryGetProperty("backend", out var be) ? be.GetString() ?? string.Empty : string.Empty;
            var lockedMethod = root.TryGetProperty("method", out var m) ? m.GetString() ?? string.Empty : string.Empty;
            var lockedTemplate = root.TryGetProperty("template", out var t) ? t.GetString() ?? string.Empty : string.Empty;

            if (!string.Equals(lockedPython, current.pythonPath, StringComparison.OrdinalIgnoreCase))
                details.Add($"pythonPath changed ({lockedPython} -> {current.pythonPath})");
            if (!string.Equals(lockedConverterEnv, current.converterEnv, StringComparison.OrdinalIgnoreCase))
                details.Add($"converterEnv changed ({lockedConverterEnv} -> {current.converterEnv})");
            if (!string.Equals(lockedBundled, current.converterBundledResolved, StringComparison.OrdinalIgnoreCase))
                details.Add($"converterBundledResolved changed ({lockedBundled} -> {current.converterBundledResolved})");
            if (!string.Equals(lockedBackend, current.backend, StringComparison.OrdinalIgnoreCase))
                details.Add($"backend changed ({lockedBackend} -> {current.backend})");
            if (!string.Equals(lockedMethod, current.method, StringComparison.OrdinalIgnoreCase))
                details.Add($"method changed ({lockedMethod} -> {current.method})");
            if (!string.Equals(lockedTemplate, current.template, StringComparison.OrdinalIgnoreCase))
                details.Add($"template changed ({lockedTemplate} -> {current.template})");

            return (details.Count > 0, details);
        }
        catch
        {
            details.Add("runtime lock parse failed");
            return (true, details);
        }
    }

    private void RaiseOllamaFtPipelineChanged()
    {
        OnPropertyChanged(nameof(OllamaFtPipelineStep));
        OnPropertyChanged(nameof(OllamaFtPipelineProgressPercent));
        OnPropertyChanged(nameof(OllamaFtPipelineProgressText));
    }

    private string ResolveDefaultOllamaBaseModelPath()
    {
        var candidate = Path.Combine(RunDirectory, "model.pt");
        return File.Exists(candidate) ? candidate : string.Empty;
    }


    private async Task<bool> TryFetchProviderRepositorySnapshotAsync(string src, GatherProviderContract provider, string sourcesDir, Stopwatch sw, CancellationToken cancellationToken)
    {
        try
        {
            if (provider.Kind == GatherProviderKind.HuggingFace && TryParseHfDatasetId(src, out var datasetId))
            {
                using var http = new HttpClient();
                var treeApi = $"https://huggingface.co/api/datasets/{datasetId}/tree/main?recursive=1";
                var treeJson = await http.GetStringAsync(treeApi, cancellationToken);
                using var treeDoc = JsonDocument.Parse(treeJson);
                var candidatePaths = new List<string>();

                if (treeDoc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var node in treeDoc.RootElement.EnumerateArray())
                    {
                        var type = node.TryGetProperty("type", out var typeEl) ? typeEl.GetString() ?? string.Empty : string.Empty;
                        var path = node.TryGetProperty("path", out var pathEl) ? pathEl.GetString() ?? string.Empty : string.Empty;
                        if (!string.Equals(type, "file", StringComparison.OrdinalIgnoreCase)) continue;
                        if (string.IsNullOrWhiteSpace(path)) continue;
                        if (!IsProviderSupportedDatasetFile(path) && !path.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase)) continue;
                        candidatePaths.Add(path);
                    }
                }

                if (candidatePaths.Count == 0)
                {
                    // Fallback: ask HF dataset-server parquet endpoint.
                    try
                    {
                        var parquetApi = $"https://datasets-server.huggingface.co/parquet?dataset={Uri.EscapeDataString(datasetId)}";
                        var parquetJson = await http.GetStringAsync(parquetApi, cancellationToken);
                        using var parquetDoc = JsonDocument.Parse(parquetJson);
                        var parquetUrls = new List<string>();
                        CollectParquetUrlsFromJsonElement(parquetDoc.RootElement, parquetUrls);
                        parquetUrls = parquetUrls
                            .Where(u => Uri.TryCreate(u, UriKind.Absolute, out var pu) && pu.AbsolutePath.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase))
                            .Distinct(StringComparer.OrdinalIgnoreCase)
                            .ToList();

                        if (parquetUrls.Count > 0)
                        {
                            var repoFolderFromParquetApi = Path.Combine(sourcesDir, $"hf_{SanitizePathToken(datasetId)}");
                            Directory.CreateDirectory(repoFolderFromParquetApi);
                            long totalBytesParquetApi = 0;
                            var totalParquetApi = parquetUrls.Count;
                            for (var i = 0; i < parquetUrls.Count; i++)
                            {
                                cancellationToken.ThrowIfCancellationRequested();
                                var parquetUrl = parquetUrls[i];
                                var outFile = Path.Combine(repoFolderFromParquetApi, $"parquet_{i:D4}.parquet");
                                await using var srcStream = await http.GetStreamAsync(parquetUrl, cancellationToken);
                                await using var dst = new FileStream(outFile, FileMode.Create, FileAccess.Write, FileShare.None);
                                await srcStream.CopyToAsync(dst, cancellationToken);
                                totalBytesParquetApi += new FileInfo(outFile).Length;
                                GatherProgressValue = 5 + ((i + 1) / (double)Math.Max(1, totalParquetApi)) * 95.0;
                                var mb = totalBytesParquetApi / (1024.0 * 1024.0);
                                var speed = mb / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                                GatherProgressText = $"Fetching source... {i + 1:N0}/{totalParquetApi:N0} files | {mb:F1} MB | {speed:F2} MB/s | elapsed {sw.Elapsed:mm\\:ss}";
                                if ((i + 1) % 10 == 0) await Task.Yield();
                            }

                            GatherStagedDatasetPath = repoFolderFromParquetApi;
                            AddGatherSource(repoFolderFromParquetApi, provider.DisplayName, ExtractLicenseLabelFromStatus(GatherLicenseText), GatherLicensePermitted);
                            GatherStatusText = $"Source repository staged (HF parquet API): {repoFolderFromParquetApi} | files={parquetUrls.Count:N0} | size={totalBytesParquetApi:N0} bytes | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                            GatherProgressText = "Fetch completed.";
                            return true;
                        }
                    }
                    catch
                    {
                        // ignored; fall through to explicit message below
                    }

                    GatherStatusText = IsEnglish
                        ? "Hugging Face dataset repository contains no supported dataset files (.jsonl/.json/.csv/.txt/.md/.parquet), and parquet API fallback returned no usable parquet URLs."
                        : "Il repository Hugging Face non contiene file dataset supportati (.jsonl/.json/.csv/.txt/.md/.parquet) e il fallback parquet API non ha restituito URL parquet utilizzabili.";
                    GatherProgressValue = 0;
                    GatherProgressText = "Idle";
                    return true;
                }

                var repoFolder = Path.Combine(sourcesDir, $"hf_{SanitizePathToken(datasetId)}");
                Directory.CreateDirectory(repoFolder);
                long totalBytes = 0;
                var total = candidatePaths.Count;
                for (var i = 0; i < candidatePaths.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var relPath = candidatePaths[i];
                    var outFile = Path.Combine(repoFolder, relPath.Replace('/', Path.DirectorySeparatorChar));
                    Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? repoFolder);
                    var encodedRel = Uri.EscapeDataString(relPath).Replace("%2F", "/");
                    var url = $"https://huggingface.co/datasets/{datasetId}/resolve/main/{encodedRel}?download=1";
                    await using var srcStream = await http.GetStreamAsync(url, cancellationToken);
                    await using var dst = new FileStream(outFile, FileMode.Create, FileAccess.Write, FileShare.None);
                    await srcStream.CopyToAsync(dst, cancellationToken);
                    totalBytes += new FileInfo(outFile).Length;
                    GatherProgressValue = 5 + ((i + 1) / (double)Math.Max(1, total)) * 95.0;
                    var mb = totalBytes / (1024.0 * 1024.0);
                    var speed = mb / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                    GatherProgressText = $"Fetching source... {i + 1:N0}/{total:N0} files | {mb:F1} MB | {speed:F2} MB/s | elapsed {sw.Elapsed:mm\\:ss}";
                    if ((i + 1) % 10 == 0) await Task.Yield();
                }

                GatherStagedDatasetPath = repoFolder;
                AddGatherSource(repoFolder, provider.DisplayName, ExtractLicenseLabelFromStatus(GatherLicenseText), GatherLicensePermitted);
                GatherStatusText = $"Source repository staged: {repoFolder} | files={candidatePaths.Count:N0} | size={totalBytes:N0} bytes | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                GatherProgressText = "Fetch completed.";
                return true;
            }

            if (provider.Kind == GatherProviderKind.GitHub && TryParseGitHubRepoId(src, out var repoId))
            {
                using var http = new HttpClient();
                http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMForgeStudio/1.0");
                var treeApi = $"https://api.github.com/repos/{repoId}/git/trees/HEAD?recursive=1";
                var treeJson = await http.GetStringAsync(treeApi, cancellationToken);
                using var treeDoc = JsonDocument.Parse(treeJson);
                var candidatePaths = new List<string>();

                if (treeDoc.RootElement.TryGetProperty("tree", out var tree) && tree.ValueKind == JsonValueKind.Array)
                {
                    foreach (var node in tree.EnumerateArray())
                    {
                        var type = node.TryGetProperty("type", out var typeEl) ? typeEl.GetString() ?? string.Empty : string.Empty;
                        var path = node.TryGetProperty("path", out var pathEl) ? pathEl.GetString() ?? string.Empty : string.Empty;
                        if (!string.Equals(type, "blob", StringComparison.OrdinalIgnoreCase)) continue;
                        if (string.IsNullOrWhiteSpace(path)) continue;
                        if (!IsProviderSupportedDatasetFile(path) && !path.EndsWith(".parquet", StringComparison.OrdinalIgnoreCase)) continue;
                        candidatePaths.Add(path);
                    }
                }

                if (candidatePaths.Count == 0)
                {
                    GatherStatusText = IsEnglish
                        ? "GitHub repository contains no supported dataset files (.jsonl/.json/.csv/.txt/.md/.parquet)."
                        : "Il repository GitHub non contiene file dataset supportati (.jsonl/.json/.csv/.txt/.md/.parquet).";
                    GatherProgressValue = 0;
                    GatherProgressText = "Idle";
                    return true;
                }

                var repoFolder = Path.Combine(sourcesDir, $"gh_{SanitizePathToken(repoId)}");
                Directory.CreateDirectory(repoFolder);
                long totalBytes = 0;
                var total = candidatePaths.Count;
                for (var i = 0; i < candidatePaths.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var relPath = candidatePaths[i];
                    var outFile = Path.Combine(repoFolder, relPath.Replace('/', Path.DirectorySeparatorChar));
                    Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? repoFolder);
                    var encodedRel = Uri.EscapeDataString(relPath).Replace("%2F", "/");
                    var url = $"https://raw.githubusercontent.com/{repoId}/HEAD/{encodedRel}";
                    await using var srcStream = await http.GetStreamAsync(url, cancellationToken);
                    await using var dst = new FileStream(outFile, FileMode.Create, FileAccess.Write, FileShare.None);
                    await srcStream.CopyToAsync(dst, cancellationToken);
                    totalBytes += new FileInfo(outFile).Length;
                    GatherProgressValue = 5 + ((i + 1) / (double)Math.Max(1, total)) * 95.0;
                    var mb = totalBytes / (1024.0 * 1024.0);
                    var speed = mb / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                    GatherProgressText = $"Fetching source... {i + 1:N0}/{total:N0} files | {mb:F1} MB | {speed:F2} MB/s | elapsed {sw.Elapsed:mm\\:ss}";
                    if ((i + 1) % 10 == 0) await Task.Yield();
                }

                GatherStagedDatasetPath = repoFolder;
                AddGatherSource(repoFolder, provider.DisplayName, ExtractLicenseLabelFromStatus(GatherLicenseText), GatherLicensePermitted);
                GatherStatusText = $"Source repository staged: {repoFolder} | files={candidatePaths.Count:N0} | size={totalBytes:N0} bytes | elapsed={sw.Elapsed.TotalSeconds:F2}s";
                GatherProgressText = "Fetch completed.";
                return true;
            }
        }
        catch (OperationCanceledException)
        {
            GatherStatusText = IsEnglish ? "Fetch canceled by user." : "Fetch annullato dall'utente.";
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return true;
        }
        catch (Exception ex)
        {
            GatherStatusText = IsEnglish ? $"Repository fetch failed: {ex.Message}" : $"Fetch repository fallito: {ex.Message}";
            GatherProgressValue = 0;
            GatherProgressText = "Idle";
            return true;
        }

        return false;
    }

    private async Task ResolveRemoteProviderLicenseAsync(string src, GatherProviderContract provider)
    {
        try
        {
            using var http = new HttpClient();
            if (provider.Kind == GatherProviderKind.HuggingFace)
            {
                if (!TryParseHfDatasetId(src, out var datasetId))
                {
                    GatherLicensePermitted = false;
                    GatherLicenseText = "License check failed: invalid Hugging Face dataset URL.";
                    return;
                }

                _gatherHfDatasetId = datasetId;
                var api = $"https://huggingface.co/api/datasets/{datasetId}";
                var json = await http.GetStringAsync(api);
                using var doc = JsonDocument.Parse(json);
                var license = ResolveLicenseFromHfApi(doc.RootElement);
                if (string.IsNullOrWhiteSpace(license))
                {
                    GatherLicensePermitted = false;
                    GatherLicenseText = $"License check failed: license not declared for {datasetId}.";
                    return;
                }

                var permitted = IsPermissiveLicense(license);
                GatherLicensePermitted = permitted;
                GatherLicenseText = permitted
                    ? $"License approved: {license} (permissive)."
                    : $"License blocked: {license} (restricted/unknown).";
                return;
            }

            if (provider.Kind == GatherProviderKind.GitHub)
            {
                if (!TryParseGitHubRepoId(src, out var repoId))
                {
                    GatherLicensePermitted = false;
                    GatherLicenseText = "License check failed: invalid GitHub repository URL.";
                    return;
                }
                var api = $"https://api.github.com/repos/{repoId}/license";
                using var req = new HttpRequestMessage(HttpMethod.Get, api);
                req.Headers.UserAgent.ParseAdd("LLMForgeStudio/1.0");
                var res = await http.SendAsync(req);
                if (!res.IsSuccessStatusCode)
                {
                    GatherLicensePermitted = false;
                    GatherLicenseText = $"License check failed: GitHub API returned {(int)res.StatusCode}.";
                    return;
                }

                var body = await res.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(body);
                var spdx = doc.RootElement.TryGetProperty("license", out var lic)
                    && lic.TryGetProperty("spdx_id", out var spdxEl)
                    ? (spdxEl.GetString() ?? string.Empty)
                    : string.Empty;
                if (string.IsNullOrWhiteSpace(spdx))
                {
                    GatherLicensePermitted = false;
                    GatherLicenseText = $"License check failed: no SPDX license found for {repoId}.";
                    return;
                }

                var permitted = IsPermissiveLicense(spdx);
                GatherLicensePermitted = permitted;
                GatherLicenseText = permitted
                    ? $"License approved: {spdx} (permissive)."
                    : $"License blocked: {spdx} (restricted/unknown).";
                return;
            }

            GatherLicensePermitted = true;
            GatherLicenseText = "No remote license resolver for provider. Manual rights confirmation required.";
        }
        catch (Exception ex)
        {
            GatherLicensePermitted = false;
            GatherLicenseText = $"License check failed: {ex.Message}";
        }
    }


    private static string ExtractTextFromJsonElement(JsonElement root)
    {
        if (root.ValueKind == JsonValueKind.Object)
        {
            if (root.TryGetProperty("prompt", out var prompt) && root.TryGetProperty("response", out var response))
                return $"{prompt.GetString()}\n{response.GetString()}".Trim();
            if (root.TryGetProperty("chosen", out var chosen) && root.TryGetProperty("rejected", out var rejected))
            {
                var p = root.TryGetProperty("prompt", out var pr) ? pr.GetString() : string.Empty;
                return $"{p}\nChosen: {chosen.GetString()}\nRejected: {rejected.GetString()}".Trim();
            }
            if (root.TryGetProperty("messages", out var messages) && messages.ValueKind == JsonValueKind.Array)
            {
                var parts = new List<string>();
                foreach (var m in messages.EnumerateArray())
                {
                    var role = m.TryGetProperty("role", out var r) ? r.GetString() : "unknown";
                    var content = m.TryGetProperty("content", out var c) ? c.GetString() : string.Empty;
                    parts.Add($"{role}: {content}");
                }
                return string.Join("\n", parts);
            }
            if (root.TryGetProperty("text", out var text) && text.ValueKind == JsonValueKind.String)
                return text.GetString() ?? string.Empty;
        }
        return root.ToString();
    }

    private static bool IsSupportedDatasetFile(string path)
        => path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".json", StringComparison.OrdinalIgnoreCase);

    private static bool IsProviderSupportedDatasetFile(string path)
        => path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
           || path.EndsWith(".json", StringComparison.OrdinalIgnoreCase);

    private static bool IsSupportedDatasetExtension(string ext)
        => string.Equals(ext, ".txt", StringComparison.OrdinalIgnoreCase)
           || string.Equals(ext, ".md", StringComparison.OrdinalIgnoreCase)
           || string.Equals(ext, ".csv", StringComparison.OrdinalIgnoreCase)
           || string.Equals(ext, ".jsonl", StringComparison.OrdinalIgnoreCase)
           || string.Equals(ext, ".json", StringComparison.OrdinalIgnoreCase);

    private static string SanitizePathToken(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "source";
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(value.Length);
        foreach (var ch in value)
            sb.Append(invalid.Contains(ch) ? '_' : ch);
        return sb.ToString().Replace('/', '_').Replace('\\', '_').Trim();
    }

    private static void CollectParquetUrlsFromJsonElement(JsonElement el, List<string> urls)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var p in el.EnumerateObject())
                {
                    if (string.Equals(p.Name, "url", StringComparison.OrdinalIgnoreCase)
                        && p.Value.ValueKind == JsonValueKind.String)
                    {
                        var v = p.Value.GetString() ?? string.Empty;
                        if (!string.IsNullOrWhiteSpace(v))
                            urls.Add(v);
                    }
                    else
                    {
                        CollectParquetUrlsFromJsonElement(p.Value, urls);
                    }
                }
                break;
            case JsonValueKind.Array:
                foreach (var item in el.EnumerateArray())
                    CollectParquetUrlsFromJsonElement(item, urls);
                break;
        }
    }

    private static string EnsureUniquePath(string requestedPath, bool isDirectory)
    {
        if (string.IsNullOrWhiteSpace(requestedPath)) return requestedPath;
        var dir = Path.GetDirectoryName(requestedPath) ?? string.Empty;
        var name = Path.GetFileNameWithoutExtension(requestedPath);
        var ext = Path.GetExtension(requestedPath);
        var candidate = requestedPath;
        var idx = 1;
        while (true)
        {
            var exists = isDirectory ? Directory.Exists(candidate) : File.Exists(candidate) || Directory.Exists(candidate);
            if (!exists) return candidate;
            var suffix = $"_{idx:D3}";
            candidate = Path.Combine(dir, $"{name}{suffix}{ext}");
            idx++;
        }
    }

    private static bool TryParseHfDatasetId(string input, out string datasetId)
    {
        datasetId = string.Empty;
        if (string.IsNullOrWhiteSpace(input)) return false;
        if (!Uri.TryCreate(input, UriKind.Absolute, out var uri)) return false;
        if (!string.Equals(uri.Host, "huggingface.co", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(uri.Host, "www.huggingface.co", StringComparison.OrdinalIgnoreCase))
            return false;
        var path = uri.AbsolutePath.Trim('/');
        var m = Regex.Match(path, @"^datasets/([^/]+/[^/]+)");
        if (!m.Success) return false;
        datasetId = m.Groups[1].Value;
        return !string.IsNullOrWhiteSpace(datasetId);
    }

    private static bool TryParseGitHubRepoId(string input, out string repoId)
    {
        repoId = string.Empty;
        if (string.IsNullOrWhiteSpace(input)) return false;
        if (!Uri.TryCreate(input, UriKind.Absolute, out var uri)) return false;
        if (!string.Equals(uri.Host, "github.com", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(uri.Host, "www.github.com", StringComparison.OrdinalIgnoreCase))
            return false;
        var parts = uri.AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2) return false;
        repoId = $"{parts[0]}/{parts[1]}";
        return true;
    }

    private static string ResolveLicenseFromHfApi(JsonElement root)
    {
        if (root.TryGetProperty("cardData", out var cardData))
        {
            if (cardData.TryGetProperty("license", out var l) && l.ValueKind == JsonValueKind.String)
                return l.GetString() ?? string.Empty;
        }
        if (root.TryGetProperty("tags", out var tags) && tags.ValueKind == JsonValueKind.Array)
        {
            foreach (var t in tags.EnumerateArray())
            {
                var s = t.GetString() ?? string.Empty;
                if (s.StartsWith("license:", StringComparison.OrdinalIgnoreCase))
                    return s.Split(':', 2)[1];
            }
        }
        return string.Empty;
    }

    private static bool IsPermissiveLicense(string license)
    {
        var s = (license ?? string.Empty).Trim().ToLowerInvariant();
        return s is "mit"
            or "apache-2.0"
            or "bsd-2-clause"
            or "bsd-3-clause"
            or "isc"
            or "mpl-2.0"
            or "cc0-1.0"
            or "unlicense";
    }

    private static (bool pass, string reason, List<object> checks, List<object> normalizedSources) EvaluateStrictMergeLicenseMatrix(IReadOnlyList<GatherSourceEntryViewModel> enabledSources)
    {
        var normalized = enabledSources
            .Select(s => new
            {
                s.Path,
                s.Provider,
                licenseRaw = s.LicenseLabel,
                licenseNormalized = NormalizeLicenseLabel(s.LicenseLabel)
            })
            .ToList();

        // Hard block unknown/restricted labels.
        var blocked = normalized.Where(s => !IsLicenseAllowedForStrictMerge(s.licenseNormalized)).ToList();
        if (blocked.Count > 0)
            return (false, "restricted/unknown/unverified license detected.", new List<object>(), normalized.Cast<object>().ToList());

        var checks = new List<object>();
        for (var i = 0; i < normalized.Count; i++)
        {
            for (var j = i + 1; j < normalized.Count; j++)
            {
                var a = normalized[i];
                var b = normalized[j];
                var pairOk = IsLicensePairCompatible(a.licenseNormalized, b.licenseNormalized);
                checks.Add(new
                {
                    sourceA = a.Path,
                    licenseA = a.licenseNormalized,
                    sourceB = b.Path,
                    licenseB = b.licenseNormalized,
                    compatible = pairOk
                });
                if (!pairOk)
                    return (false, $"license matrix incompatible pair: {a.licenseNormalized} vs {b.licenseNormalized}.", checks, normalized.Cast<object>().ToList());
            }
        }

        return (true, "ok", checks, normalized.Cast<object>().ToList());
    }

    private static string NormalizeLicenseLabel(string? raw)
    {
        var s = (raw ?? string.Empty).Trim().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(s)) return "unknown";
        if (s.Contains("manual-ack", StringComparison.OrdinalIgnoreCase)) return "manual-ack";
        if (s.Contains("derived", StringComparison.OrdinalIgnoreCase)) return "derived";
        if (s.Contains("restricted", StringComparison.OrdinalIgnoreCase)) return "restricted";
        if (s.Contains("unknown", StringComparison.OrdinalIgnoreCase)) return "unknown";
        if (s.StartsWith("license:", StringComparison.OrdinalIgnoreCase))
            s = s.Split(':', 2)[1].Trim();
        return s;
    }

    private static bool IsLicenseAllowedForStrictMerge(string normalized)
    {
        // Non-overridable strict block list for M12 compliance gate.
        if (normalized is "unknown" or "restricted" or "manual-ack") return false;
        if (normalized == "derived") return true;
        return IsPermissiveLicense(normalized);
    }

    private static bool IsLicensePairCompatible(string a, string b)
    {
        // In strict mode, only permissive/derived labels may mix.
        if (!IsLicenseAllowedForStrictMerge(a) || !IsLicenseAllowedForStrictMerge(b))
            return false;
        return true;
    }

    private static string RunCommand(string fileName, string arguments)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var process = Process.Start(psi);
            if (process is null) return string.Empty;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(2500);
            return output;
        }
        catch
        {
            return string.Empty;
        }
    }
}

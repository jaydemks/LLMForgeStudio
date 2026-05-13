namespace LLMForgeStudio.App.Core.Training;

public sealed record TrainingLogEntry(int Step, double TrainLoss, double ValLoss, double TokensPerSecond, string Message);

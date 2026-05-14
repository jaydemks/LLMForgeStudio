using System.Text.Json;

namespace LLMForgeStudio.App.Core.Diagnostics;

public sealed class UiDebugLogger
{
    private readonly SemaphoreSlim _mutex = new(1, 1);
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = false };

    public async Task WriteAsync(string runDirectory, string eventType, string message, object? data = null)
    {
        try
        {
            Directory.CreateDirectory(runDirectory);
            var path = Path.Combine(runDirectory, "ui_debug_log.jsonl");
            var evt = new UiDebugEvent
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                EventType = eventType,
                Message = message,
                Data = data
            };

            var line = JsonSerializer.Serialize(evt, _jsonOptions) + Environment.NewLine;
            await _mutex.WaitAsync();
            try
            {
                await File.AppendAllTextAsync(path, line);
            }
            finally
            {
                _mutex.Release();
            }
        }
        catch
        {
            // Never break the UX flow because of debug logging.
        }
    }

    private sealed class UiDebugEvent
    {
        public DateTimeOffset TimestampUtc { get; init; }
        public string EventType { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public object? Data { get; init; }
    }
}

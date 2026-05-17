namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private async Task GatherHandoffToScratchAsync()
    {
        var staged = GatherStagedDatasetPath;
        if (string.IsNullOrWhiteSpace(staged))
        {
            GatherStatusText = "No staged dataset to handoff.";
            return;
        }
        await RunBusyOverlayAsync(
            IsEnglish ? "Handing off dataset..." : "Passaggio dataset in corso...",
            IsEnglish ? "Preparing Dataset section safely for large inputs." : "Preparazione sezione Dataset in modo sicuro per input grandi.",
            async cancellationToken =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (File.Exists(staged))
                    await ImportDatasetFromPathAsync(staged);
                else if (Directory.Exists(staged))
                    await ImportDatasetFromFolderAsync(staged);
                cancellationToken.ThrowIfCancellationRequested();
                SelectedSection = "Dataset";
                GatherStatusText = "Handoff to from-scratch pipeline completed.";
            });
    }

    private void GatherHandoffToFineTune()
    {
        if (string.IsNullOrWhiteSpace(GatherStagedDatasetPath))
        {
            GatherStatusText = "No staged dataset to handoff.";
            return;
        }
        _ = RunBusyOverlayAsync(
            IsEnglish ? "Handing off dataset..." : "Passaggio dataset in corso...",
            IsEnglish ? "Updating Fine-Tuning dataset path." : "Aggiornamento percorso dataset per Fine-Tuning.",
            cancellationToken =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                OllamaFtDatasetPath = GatherStagedDatasetPath;
                SelectedSection = "Fine-Tuning (Ollama, Experimental)";
                GatherStatusText = "Handoff to fine-tuning pipeline completed.";
                return Task.CompletedTask;
            });
    }

    private async Task RunBusyOverlayAsync(string title, string subText, Func<CancellationToken, Task> action)
    {
        _busyOverlayCts?.Cancel();
        _busyOverlayCts?.Dispose();
        _busyOverlayCts = new CancellationTokenSource();
        var token = _busyOverlayCts.Token;
        BusyOverlayText = title;
        BusyOverlaySubText = subText;
        BusyOverlayCancelable = true;
        BusyOverlayOpacity = 0;
        ShowBusyOverlay = true;
        try
        {
            await Task.Yield();
            BusyOverlayOpacity = 1;
            await action(token);
        }
        catch (OperationCanceledException)
        {
            BusyOverlaySubText = IsEnglish ? "Operation canceled by user." : "Operazione annullata dall'utente.";
        }
        finally
        {
            BusyOverlayOpacity = 0;
            await Task.Delay(180);
            ShowBusyOverlay = false;
            BusyOverlayCancelable = false;
            _busyOverlayCts?.Dispose();
            _busyOverlayCts = null;
        }
    }

    private void CancelBusyOverlay()
    {
        if (!BusyOverlayCancelable) return;
        _busyOverlayCts?.Cancel();
    }
}

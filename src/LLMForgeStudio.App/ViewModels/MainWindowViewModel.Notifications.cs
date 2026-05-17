namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private void ToggleNotificationsOverlay()
    {
        ShowNotificationsOverlay = !ShowNotificationsOverlay;
    }

    private void ClearNotifications()
    {
        Notifications.Clear();
        UnreadNotificationsCount = 0;
        OnPropertyChanged(nameof(NotificationsButtonText));
    }

    private void AddNotification(string level, string title, string message)
    {
        var entry = new NotificationEntryViewModel
        {
            Level = level,
            Title = title,
            Message = message,
            TimestampLocal = DateTime.Now,
            IsUnread = !ShowNotificationsOverlay
        };

        Notifications.Insert(0, entry);
        while (Notifications.Count > 200)
            Notifications.RemoveAt(Notifications.Count - 1);

        if (!ShowNotificationsOverlay)
            UnreadNotificationsCount++;
    }
}

public sealed class NotificationEntryViewModel : ObservableObject
{
    private bool _isUnread;

    public DateTime TimestampLocal { get; init; }
    public string Level { get; init; } = "info";
    public string Title { get; init; } = string.Empty;
    public string Message { get; init; } = string.Empty;

    public bool IsUnread
    {
        get => _isUnread;
        set => SetProperty(ref _isUnread, value);
    }

    public string TimestampText => TimestampLocal.ToString("yyyy-MM-dd HH:mm:ss");
}


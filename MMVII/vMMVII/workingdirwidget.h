#ifndef WORKINGDIRWIDGET_H
#define WORKINGDIRWIDGET_H

#include <QWidget>

namespace Ui {
class WorkingDirWidget;
}

class WorkingDirWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WorkingDirWidget(bool hasCommand, QWidget *parent = nullptr);
    ~WorkingDirWidget();

    void setLocked(bool lock);

public slots:
    void workingDirCleared();
    void removeCurrentFromList();

signals:
    void workingDirChanged();
    void logSignal();

private slots:
    void addDir(const QString &dir);
    bool addFromHistory(const QString &dir);
    void workingDirSelected(const QString&);
    void openDir();
    void workingDirContextMenu(const QPoint& pos);

private:
    void selectDir();

    bool firstNotInHistory;
    Ui::WorkingDirWidget *ui;
};

#endif // WORKINGDIRWIDGET_H

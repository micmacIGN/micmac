#include "workingdirwidget.h"
#include "ui_workingdirwidget.h"
#include "global.h"
#include "settings.h"

#include <QFileDialog>
#include <QDir>
#include <QLineEdit>
#include <QMessageBox>
#include <QStylePainter>
#include <QMenu>
#include <QDesktopServices>


WorkingDirWidget::WorkingDirWidget(bool hasCommand, QWidget *parent) :
    QWidget(parent),
    firstNotInHistory(false),
    ui(new Ui::WorkingDirWidget)
{
    ui->setupUi(this);
    ui->workingDirCB->setDuplicatesEnabled(false);

    auto dirs = Settings::workingDirs();
    QStringList newDirs;
    for (const auto& d:dirs) {
        if (QDir(d).exists() && addFromHistory(d))
            newDirs.append(d);
    }
    Settings::setWorkingDirs(newDirs);

    auto curdir = QDir::current();
    if (curdir.exists(MMVII_LOG_FILE) || hasCommand) {
        addDir(curdir.canonicalPath());
    } else {
        if (ui->workingDirCB->findText(curdir.canonicalPath()) < 0)
            ui->workingDirCB->addItem(curdir.canonicalPath());
    }

    for (int i=0; i<ui->workingDirCB->count(); i++) {
        if (QDir::setCurrent(ui->workingDirCB->itemText(i))) {
            ui->workingDirCB->setCurrentIndex(i);
            break;
        }
    }

    connect(ui->workingDirPB, &QPushButton::clicked, this, &WorkingDirWidget::selectDir);
    connect(ui->workingDirCB, &QComboBox::currentTextChanged, this, &WorkingDirWidget::workingDirSelected);
    connect(ui->browseCB, &QPushButton::clicked, this, &WorkingDirWidget::openDir);
    connect(ui->logPB, &QPushButton::clicked, this, &WorkingDirWidget::logSignal);
    ui->workingDirCB->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(ui->workingDirCB, &QComboBox::customContextMenuRequested, this, &WorkingDirWidget::workingDirContextMenu);
}

WorkingDirWidget::~WorkingDirWidget()
{
    delete ui;
}


void WorkingDirWidget::setLocked(bool lock)
{
    ui->workingDirPB->setDisabled(lock);
    ui->workingDirCB->setDisabled(lock);
}

void WorkingDirWidget::addDir(const QString &dir)
{
    ui->workingDirCB->blockSignals(true);
    if (ui->workingDirCB->count() > 0 && firstNotInHistory) {
        ui->workingDirCB->removeItem(0);
    }
    ui->workingDirCB->blockSignals(false);
    firstNotInHistory = false;
    for (int i = 0; i < ui->workingDirCB->count(); i++) {
        if (dir == ui->workingDirCB->itemText(i)) {
            ui->workingDirCB->setCurrentIndex(i);
            return;
        }
    }
    ui->workingDirCB->insertItem(0,dir);
    ui->workingDirCB->setCurrentIndex(0);
    firstNotInHistory = true;
}

bool WorkingDirWidget::addFromHistory(const QString &dir)
{
    if (ui->workingDirCB->findText(dir) >= 0)
        return false;

    ui->workingDirCB->insertItem(0,dir);
    ui->workingDirCB->setCurrentIndex(0);
    return true;
}

void WorkingDirWidget::workingDirSelected(const QString &newDir)
{
    if (QFileInfo(newDir).absoluteFilePath() == QDir::currentPath())
        return;
    if (! QDir::setCurrent(newDir)) {
        addDir(QDir::currentPath());
        QMessageBox::warning(this, QGuiApplication::applicationDisplayName(),
                             tr("Cannot change current directory to '%1'").arg(newDir)
                         );
        return;
    }
    emit workingDirChanged();
}

void WorkingDirWidget::workingDirCleared()
{
    int current = ui->workingDirCB->currentIndex();
    for (int i=0; i<current; i++)
        ui->workingDirCB->removeItem(0);
    while (ui->workingDirCB->count() > 1)
        ui->workingDirCB->removeItem(1);
}


void WorkingDirWidget::selectDir()
{
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Select the working directory"));
    if (dirName.isEmpty())
        return;
    addDir(dirName);
}


void WorkingDirWidget::openDir()
{
    QDesktopServices::openUrl( QUrl::fromLocalFile(QDir::currentPath()) );
}

void WorkingDirWidget::workingDirContextMenu(const QPoint &pos)
{
    QPoint globalPos = ui->workingDirCB->mapToGlobal(pos);
    QMenu menu;
    QAction *action = menu.addAction("&Remove from list",this, SLOT(removeCurrentFromList()));
    if (ui->workingDirCB->count() <= 1)
        action->setEnabled(false);
    menu.exec(globalPos);
}


void WorkingDirWidget::removeCurrentFromList()
{
    if (ui->workingDirCB->count() > 1) {
        ui->workingDirCB->removeItem(ui->workingDirCB->currentIndex());
        QStringList dirs;
        for (int i=0; i<ui->workingDirCB->count(); i++)
            dirs.append(ui->workingDirCB->itemText(i));
        Settings::setWorkingDirs(dirs);
    }
}



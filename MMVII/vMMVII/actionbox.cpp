#include "actionbox.h"
#include "ui_actionbox.h"

ActionBox::ActionBox(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ActionBox)
{
    ui->setupUi(this);

    statusBar = new QStatusBar(this);
    ui->verticalLayout->insertWidget(0,statusBar);

    connect(ui->quitButton,&QPushButton::clicked,this,&ActionBox::quitSignal);
    connect(ui->settingsButton,&QPushButton::clicked,this,&ActionBox::settingsSignal);
    connect(ui->backButton,&QPushButton::clicked,this,&ActionBox::backSignal);
    connect(ui->clearButton,&QPushButton::clicked,this,&ActionBox::clearSignal);
    connect(ui->editButton,&QPushButton::clicked,this,&ActionBox::editSignal);
    connect(ui->runSelectedButton,&QPushButton::clicked,this,&ActionBox::runSelectedSignal);
    connect(ui->runEditedButton,&QPushButton::clicked,this,&ActionBox::runEditedSignal);
}

ActionBox::~ActionBox()
{
    delete ui;
}

void ActionBox::setCommandSelection(bool commandSelection)
{
    ui->editButton->setVisible(commandSelection);
    ui->runSelectedButton->setVisible(commandSelection);

    ui->backButton->setVisible(!commandSelection);
    ui->clearButton->setVisible(!commandSelection);
    ui->runEditedButton->setVisible(!commandSelection);
}

void ActionBox::setStatusMessage(const QString &msg)
{
    statusBar->showMessage(msg,0);
}

void ActionBox::runEnabled(bool on)
{
    ui->runEditedButton->setEnabled(on);
    ui->runSelectedButton->setEnabled(on);
}

void ActionBox::editEnabled(bool on)
{
    ui->editButton->setEnabled(on);
}


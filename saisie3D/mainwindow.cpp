#include <QLayout>
#include <QFileDialog>
#include <QMessageBox>

#include "mainwindow.h"
#include "ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    m_glWidget( NULL )
{
    ui->setupUi(this);

    m_glWidget = new GLWidget;

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(m_glWidget);

    ui->OpenglLayout->setLayout(layout);

    connect(m_glWidget,	SIGNAL(filesDropped(const QStringList&)), this,	SLOT(addFiles(const QStringList&)));

    connect(m_glWidget,	SIGNAL(mouseWheelRotated(float)),			this,       SLOT(echoMouseWheelRotate(float)));

    //"Points selection" menu
    connect(ui->actionTogglePoints_selection, SIGNAL(toggled(bool)), this, SLOT(togglePointsSelection(bool)));


    connectActions();
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::checkForLoadedEntities()
{
    bool loadedEntities = true;
    m_glWidget->displayNewMessage(QString(), GLWidget::SCREEN_CENTER_MESSAGE); //clear (any) message in the middle area

    if (!m_glWidget->hasCloudLoaded())
    {
        m_glWidget->displayNewMessage("Drag & drop files on the window to load them!", GLWidget::SCREEN_CENTER_MESSAGE);
        loadedEntities = false;
    }

    return loadedEntities;
}

void MainWindow::addFiles(const QStringList& filenames)
{
    for (int i=0;i<filenames.size();++i)
    {
        m_glWidget->addPly(filenames[i]);
        m_glWidget->updateGL();
    }

    checkForLoadedEntities();
}

void MainWindow::toggleFullScreen(bool state)
{
    if (state)
        showFullScreen();
    else
        showNormal();
    m_glWidget->updateGL();
}

void MainWindow::togglePointsSelection(bool state)
{
    /*QMessageBox msgBox;
    QString text = "Point selection on\n";
    msgBox.setText(text);
    msgBox.exec();/**/

    if (state)
        m_glWidget->setInteractionMode(GLWidget::SEGMENT_POINTS);
    else
        m_glWidget->setInteractionMode(GLWidget::TRANSFORM_CAMERA);
}

void MainWindow::doActionDisplayShortcuts()
{
    QMessageBox msgBox;
    QString text;
    text += "Shortcuts:\n\n";
    text += "F11: Toggle full screen\n";
    text += "\n";
    text += "F5 : Toggle rotation mode / selection mode\n";
    text += "    - left click : add a point to polyline ";
    text += "    - right click: close polyline\n";
    text += "    - escape: delete polyline\n";
    text += "    - space bar: keep points inside polyline\n";
    text += "    - delete key: keep points outside polyline\n";
    msgBox.setText(text);
    msgBox.exec();
}

void MainWindow::connectActions()
{
    connect(ui->actionFullScreen, SIGNAL(toggled(bool)), this, SLOT(toggleFullScreen(bool)));

    connect(ui->actionHelpShortcuts, SIGNAL(triggered()), this, SLOT(doActionDisplayShortcuts()));

    connect(ui->actionSetViewTop,					SIGNAL(triggered()),						this,	SLOT(setTopView()));
    connect(ui->actionSetViewBottom,				SIGNAL(triggered()),						this,	SLOT(setBottomView()));
    connect(ui->actionSetViewFront,					SIGNAL(triggered()),						this,	SLOT(setFrontView()));
    connect(ui->actionSetViewBack,					SIGNAL(triggered()),						this,	SLOT(setBackView()));
    connect(ui->actionSetViewLeft,					SIGNAL(triggered()),						this,	SLOT(setLeftView()));
    connect(ui->actionSetViewRight,					SIGNAL(triggered()),						this,	SLOT(setRightView()));
}

void MainWindow::setTopView()
{
    m_glWidget->setView(MM_TOP_VIEW);
}

void MainWindow::setBottomView()
{
    m_glWidget->setView(MM_BOTTOM_VIEW);
}

void MainWindow::setFrontView()
{
    m_glWidget->setView(MM_FRONT_VIEW);
}

void MainWindow::setBackView()
{
    m_glWidget->setView(MM_BACK_VIEW);
}

void MainWindow::setLeftView()
{
    m_glWidget->setView(MM_LEFT_VIEW);
}

void MainWindow::setRightView()
{
    m_glWidget->setView(MM_RIGHT_VIEW);
}

void MainWindow::echoMouseWheelRotate(float wheelDelta_deg)
{
   // if (checkBoxCameraLink->checkState() != Qt::Checked)
      //  return;

    GLWidget* sendingWindow = dynamic_cast<GLWidget*>(sender());
    if (!sendingWindow)
        return;

    sendingWindow->onWheelEvent(wheelDelta_deg);
}

#include "WorkbenchWidget.h"
#include "ui_WorkbenchWidget.h"

cWorkBenchWidget::cWorkBenchWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::cWorkBenchWidget),
    _dIOCamera(NULL)
{
    ui->setupUi(this);

    QPushButton* bImages = ui->buttonImages;

    connect(bImages,SIGNAL(clicked()),this,SLOT(chooseImages()));

    connect(listViewImages(),SIGNAL(clicked(QModelIndex)),this,SLOT(updateOrientation(QModelIndex)));
    connect(listViewImages(),SIGNAL(clicked(QModelIndex)),this,SLOT(updateTiePoint(QModelIndex)));

    _filters_DIR_Ori << "Ori-*" << "ORI-*" << "ori-*";

}

cWorkBenchWidget::~cWorkBenchWidget()
{
    delete ui;
}

void cWorkBenchWidget::initModelFileImages()
{
    _modelFileImage = new QFileSystemModel(this);
    _modelFileImage->setFilter( QDir::NoDotAndDotDot | QDir::Files );
    _modelFileImage->setNameFilters(_filesImages);
    _modelFileImage->setNameFilterDisables(false);
    _modelFileImage->setRootPath(_mainDir.absolutePath());
}

QComboBox*cWorkBenchWidget::comboB_Orientations(){return ui->comboBox_Orientation;}

QListView*cWorkBenchWidget::listViewImages(){return ui->listWidget_Images;}

QLineEdit*cWorkBenchWidget::lineImages(){return ui->lineImages;}

QLineEdit*cWorkBenchWidget::lineMainDir(){return ui->lineEdit_MainDirectory;}

QTextEdit*cWorkBenchWidget::textEditOrient(){ return ui->textEdit_Orientation;}

QLabel* cWorkBenchWidget::labeImage(){ return ui->labelImage;}

void cWorkBenchWidget::updateTiePoint(const QModelIndex & index)
{
    QDir pastisFolder(_mainDir);

    QFileInfo fileImage(_mainDir,index.data().toString());

    QFileInfo fileTiePoint;
    QPolygonF poly;

    if(pastisFolder.cd("Pastis"))
    {
        QFileInfoList	files = pastisFolder.entryInfoList(QStringList() << QString("*") + fileImage.baseName() + QString("*"), QDir::NoDotAndDotDot | QDir::Files);
        int				scale = 1e9;

        for (int i = 0; i < files.size(); ++i)
        {
            QRegExp rx("*.dat");
            rx.setPatternSyntax(QRegExp::Wildcard);
            if(rx.exactMatch(files[i].fileName()))
            {
                QRegExp rxlen("(\\d+)(?:\\s*)(_Teta)");
                int pos = rxlen.indexIn(files[i].baseName());
                if (pos > -1)
                {
                    int scaleFile = rxlen.cap(1).toInt();
                    if(scaleFile < scale)
                    {
                        fileTiePoint = files[i];
                        scale = scaleFile;
                    }
                }
            }
        }
        if(fileTiePoint.exists())
            _dIOTieFile->load(fileTiePoint.filePath(),poly);

    }
    if(fileImage.exists())
    {
        QImage image;
        image.load(fileImage.filePath());

        if(image.isNull()) //TODO Attention fuite memoire
            image = *_dIOImage->loadImage(fileImage.filePath(),false);

        QPixmap pixmap = QPixmap::fromImage(image);
        QPainter painter(&pixmap);
        painter.setPen(Qt::red);

        painter.drawPoints(poly);

        labeImage()->setPixmap(pixmap.scaledToWidth(min(labeImage()->width(),image.width())));
    }
}
deviceIOImage* cWorkBenchWidget::dIOImage() const
{
    return _dIOImage;
}

void cWorkBenchWidget::setDIOImage(deviceIOImage* dIOImage)
{
    _dIOImage = dIOImage;
}

deviceIOTieFile* cWorkBenchWidget::dIOTieFile() const
{
    return _dIOTieFile;
}

void cWorkBenchWidget::setDIOTieFile(deviceIOTieFile* dIOTieFile)
{
    _dIOTieFile = dIOTieFile;
}


void cWorkBenchWidget::updateOrientation(const QModelIndex & index)
{

    QFileInfo fileOri(_mainDir.absolutePath() + _mainDir.separator() + QString("Ori-") + comboB_Orientations()->currentText() +  _mainDir.separator() + QString("Orientation-") + index.data().toString() + QString(".xml"));

    QString txtOrientation(tr("No orientation file"));

    if(_dIOCamera && fileOri.exists())
    {
        cCamHandler* cam = _dIOCamera->loadCamera(fileOri.absoluteFilePath());

        QString pos;
        &pos << cam->getCenter();

        QString rot;
        &rot << cam->getRotation();

        QString posrot = tr("Position : ") + pos + tr(" Rotation : ") + rot;

        txtOrientation = QString(tr("Orientation in ") + comboB_Orientations()->currentText() +  tr(" image ")) + index.data().toString() + "\n" +  posrot;
    }

    textEditOrient()->clear();
    textEditOrient()->setText(txtOrientation);
}

void cWorkBenchWidget::chooseImages()
{

    QStringList dirAndfiles = QFileDialog::getOpenFileNames(this,tr("Select images"),"/home/gchoqueux/Documents/Aerien-Cuxha",tr("Images (*.png *.PNG *.jpg *.JPG *.TIF *.tif *.cr2 *.CR2 *.crw *.CRW *.nef *.NEF);;All Files (*.*)"));

    if (dirAndfiles.size())
    {
        _mainDir = QFileInfo(dirAndfiles[0]).dir();

        QString		sDir	= _mainDir.absolutePath() +  QString("/");
        _filesImages.clear();

        _filesImages	= dirAndfiles.replaceInStrings(sDir,"");

        lineImages()->setText( _filesImages.join("|"));
        lineMainDir()->setText(_mainDir.absolutePath());

        _oriDirectory.clear();
        _oriDirectory = _mainDir.entryList(_filters_DIR_Ori,QDir::Dirs);

        comboB_Orientations()->clear();
        comboB_Orientations()->addItems(_oriDirectory.replaceInStrings("Ori-",""));

        initModelFileImages();

        listViewImages()->setModel(_modelFileImage);
        listViewImages()->setRootIndex(_modelFileImage->index(_mainDir.absolutePath()));

    }
}
deviceIOCamera* cWorkBenchWidget::dIOCamera() const
{
    return _dIOCamera;
}

void cWorkBenchWidget::setDIOCamera(deviceIOCamera* dIOCamera)
{
    _dIOCamera = dIOCamera;
}



//TODO: ELISE HEADER?
#if ELISE_QT
#include "general/visual_mainwindow.h"

#include "QT_interface_Elise.h"

static int IntMin = -1e9;
static int IntMax =  1e9;
static double DoubleMin = -1e10;
static double DoubleMax =  1e10;

bool isFirstArgMalt(string val)
{
    list <string> strList = ListOfVal(eTMalt_NbVals,"eTMalt_");
    if (std::find(strList.begin(), strList.end(), val) != strList.end())
        return true;
    return false;
}

// aVAM: Mandatory args
// aVAO: Optional args

visual_MainWindow::visual_MainWindow(vector<cMMSpecArg> & aVAM,
                                     vector<cMMSpecArg> & aVAO,
                                     string aFirstArg,
                                     QString aLastDir,
                                     QWidget *parent
                                     ):
    QWidget(parent),
    mlastDir(aLastDir),
    mFirstArg(aFirstArg),
    _SaisieWin(new SaisieQtWindow(BOX2D)),
    _showDialog(false),
    _bMaltGeomImg(false)
{
    _SaisieWin->setDevIOCamera((deviceIOCamera*)new deviceIOCameraElise);
    _SaisieWin->setDevIOImage((deviceIOImageElise*)new deviceIOImageElise);

    _SaisieWin->setBanniere(QString(getBanniereMM3D().c_str()));
    _SaisieWin->setGit_revision(QString(gitRevision().c_str()));

    setWindowFlags(Qt::WindowStaysOnTopHint);
    //setAttribute( Qt::WA_DeleteOnClose );

    QVBoxLayout *verticalLayout = new QVBoxLayout(this);

    setLayout(verticalLayout);

    toolBox = new QToolBox();

    verticalLayout->addWidget(toolBox);

    moveArgs(aVAM, aVAO);

    addGridLayout(aVAM, tr("&Mandatory arguments"));

    if (aVAO.size())
    {
        addGridLayout(aVAO, tr("&Optional arguments"), false);

        connect(toolBox, SIGNAL(currentChanged(int)), this, SLOT(_adjustSize(int)));
    }

    runCommandButton = new QPushButton(tr(" &Run command "), this);
    showPromptDialog = new QCheckBox(tr("Show dialog when job done"), this);
    showPromptDialog->setChecked(false);

    QSpacerItem *vSpacer = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    QGridLayout* RunGridLayout = new QGridLayout();
    RunGridLayout->addItem(vSpacer, 0, 0);
    RunGridLayout->addWidget(showPromptDialog, 1, 0);
    QSpacerItem* spacer = new QSpacerItem(0,0,QSizePolicy::Expanding, QSizePolicy::Fixed);
    RunGridLayout->addItem(spacer, 1, 1);
    RunGridLayout->addWidget(runCommandButton, 1, 2);
    verticalLayout->insertLayout(1, RunGridLayout);

    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));
    connect(showPromptDialog,SIGNAL(stateChanged(int)),this,SLOT(setShowDialog(int)));

    //shortcut quit
    QKeySequence ks(Qt::CTRL + Qt::Key_Q);
    QShortcut* shortcut = new QShortcut(ks, this);
    QObject::connect(shortcut, SIGNAL(activated()), this, SLOT(close()));
}

visual_MainWindow::~visual_MainWindow()
{
    delete toolBox;
    delete runCommandButton;
    delete showPromptDialog;
}

void visual_MainWindow::moveArgs(vector<cMMSpecArg> &aVAM, vector<cMMSpecArg> &aVAO)
{
    for (int aK=0; aK < (int) aVAM.size(); aK++)
    {
        cMMSpecArg arg = aVAM[aK];

        if (( arg.Type() == AMBT_string ) && (*(arg.DefaultValue<string>()) == "GeomImage")) _bMaltGeomImg = true;
        if (_bMaltGeomImg) break;
    }

    if (_bMaltGeomImg)
    {
        int idx = -1;
        for (int aK=0; aK < (int) aVAO.size(); aK++)
        {
            if (aVAO[aK].NameArg() == "Master")
            {
                aVAM.push_back(aVAO[aK]);
                idx = aK;
            }
        }

        if (idx >= 0)
        {
            for (int aK = idx; aK < (int) aVAO.size()-1; aK++)
                aVAO[aK] = aVAO[aK+1];

            aVAO.pop_back();
        }
    }

    //Remove arg with flag "for internal use"
    bool bHasBox2D = false;
    for (int aK=0; aK < (int) aVAO.size(); aK++)
    {
        cMMSpecArg arg = aVAO[aK];

        if (arg.IsForInternalUse())
        {
            aVAO.erase(aVAO.begin() + aK);
            aK--;
        }
        else if (( arg.Type() ==  AMBT_Box2di ) || ( arg.Type() ==  AMBT_Box2dr ) ) bHasBox2D = true;
    }

    //set minimum width
    if  (toolBox != NULL)
    {
        if (bHasBox2D) toolBox->setMinimumWidth(800);
        else toolBox->setMinimumWidth(470);
    }

    //Sort optional args
    cCmpMMSpecArg aCmpMMSpecArg;
    std::sort(aVAO.begin(),aVAO.end(),aCmpMMSpecArg);
}

void visual_MainWindow::addGridLayout(const vector<cMMSpecArg> &aVA, QString pageName, bool addSpace)
{
    QWidget* mPage = new QWidget();

    toolBox->addItem(mPage, pageName);

    QGridLayout* gridLayout = new QGridLayout(mPage);

    buildUI(aVA, gridLayout, mPage);

    if (addSpace)
    {
        gridLayout->addItem(new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding), (int)aVA.size(), 0);
    }
}

void visual_MainWindow::buildUI(const vector<cMMSpecArg>& aVA, QGridLayout *layout, QWidget *parent)
{
    for (int aK=0 ; aK<int(aVA.size()) ; aK++)
    {
        cMMSpecArg aArg = aVA[aK];
        //cout << "arg " << aK << " ; Type is " << aArg.NameType() << " ; Name is " << aArg.NameArg() <<  endl;

        add_label(layout, parent, aK, aArg);

        if (!aArg.IsInit() && aArg.IsOpt())
        {
             add_select(layout, parent, aK, aArg);
        }
        else if (aArg.IsBool()) // because some boolean values are set with int
        {
            add_combo(layout, parent, aK, aArg);
        }
        else
        {
            switch (aArg.Type())
            {
            case AMBT_Box2di:
                add_4i_SpinBox(layout, parent, aK, aArg);
                break;
            case AMBT_Box2dr:
                add_4d_SpinBox(layout, parent, aK, aArg);
                break;
            case AMBT_bool:
                add_combo(layout, parent, aK, aArg);
                break;
            case AMBT_INT:
            case AMBT_U_INT1:
            case AMBT_INT1:
                add_1i_SpinBox(layout, parent, aK, aArg);
                break;
            case AMBT_REAL:
                add_1d_SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt2di:
                add_2i_SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt2dr:
                add_2d_SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt3dr:
                add_3d_SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt3di:
                add_3i_SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_string:
            {
                if (!aArg.EnumeratedValues().empty())
                    add_combo(layout, parent, aK, aArg);
                else
                    add_select(layout, parent, aK, aArg);
                break;
            }
            case AMBT_vector_string:
            case AMBT_vector_Pt2dr:
            case AMBT_vector_int:
            case AMBT_vector_double:
            case AMBT_vvector_int:
                add_select(layout, parent, aK, aArg);
                break;
            case AMBT_char:
            case AMBT_unknown:
                cout << "type non gere: " << aArg.NameType() << endl;
                break;
            }
        }
        //ShowEnum(aVA[aK]);
    }
}

bool visual_MainWindow::getSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar)
{
    int val = ((QSpinBox*) aIn->Widgets()[aK].second)->value();

    stringstream ss;
    ss << val;
    aAdd += ss.str() + endingCar;

    return aIn->Arg().IsDefaultValue<int>(val);
}

bool visual_MainWindow::getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar)
{
    double val = ((QDoubleSpinBox*) aIn->Widgets()[aK].second)->value();

    stringstream ss;
    ss << val;
    aAdd += ss.str() + endingCar;

    return  aIn->Arg().IsDefaultValue<double>(val);
}

void visual_MainWindow::saveSettings()
{
    QSettings settings(QApplication::organizationName(), QApplication::applicationName());

    settings.beginGroup("FilePath");
    settings.setValue("Path", mlastDir);
    settings.endGroup();
}

void visual_MainWindow::onRunCommandPressed()
{
    saveSettings();

    bool runCom = true;

    string aCom = MM3dBinFile(argv_recup) + " " + mFirstArg + " ";

    for (unsigned int aK=0;aK<vInputs.size();aK++)
    {
        string aAdd;

        cInputs* aIn = vInputs[aK];

        switch(aIn->Type())
        {
        case eIT_LineEdit:
        {
            if (aIn->Widgets().size() == 1)
            {
                QLineEdit* lEdit = (QLineEdit*) aIn->Widgets()[0].second;

                QString txt = lEdit->text().simplified();

                if (aIn->Arg().IsExistFileWithRelativePath())
                {
                    txt = QFileInfo(txt).fileName();
                }

                if (!txt.isEmpty()  || isFirstArgMalt(txt.toStdString()))
                {
                    aAdd += QUOTE(txt.toStdString());
                }
                else if (!aIn->IsOpt()) runCom = false;
            }
            break;
        }
        case eIT_ComboBox:
        {
            if (aIn->Widgets().size() == 1)
            {
                QString aStr = ((QComboBox*) aIn->Widgets()[0].second)->currentText(); //warning

                if(aIn->Arg().IsBool() || (aIn->Arg().Type() == AMBT_bool))
                {
                    bool aB = false;

                    if (aStr == "True")
                    {
                        aStr = "1";
                        aB = true;
                    }
                    else
                        aStr = "0";

                    //check if value is different from default value
                    if ( !aIn->Arg().IsDefaultValue<bool>(aB) )
                        aAdd += aStr.toStdString();
                }
                else if (( aIn->Arg().IsOpt() && !aIn->Arg().IsDefaultValue<string>(aStr.toStdString()) ) || !aIn->Arg().IsOpt())
                {
                    aAdd += aStr.toStdString();
                }
            }
            break;
        }
        case eIT_SpinBox:
        {
            if (aIn->Widgets().size() == 1)
            {
                if (getSpinBoxValue(aAdd, aIn, 0)) aAdd.clear();
            }
            else
            {
                string toAdd = "[";
                int nbDefVal = 0;

                int max = aIn->NbWidgets()-1;

                for (int aK=0; aK < max;++aK)
                {
                    if ( getSpinBoxValue(toAdd, aIn, aK, ",") ) nbDefVal++;
                }

                if ( getSpinBoxValue(toAdd, aIn, max, "]") ) nbDefVal++;

                if (nbDefVal < aIn->NbWidgets()) aAdd += toAdd;
            }
            break;
        }
        case eIT_DoubleSpinBox:
        {
            if (aIn->Widgets().size() == 1)
            {
                if (getDoubleSpinBoxValue(aAdd, aIn, 0)) aAdd.clear();
            }
            else
            {
                string toAdd = "[";
                int nbDefVal = 0;

                int max = aIn->NbWidgets()-1;

                for (int aK=0; aK < max ;++aK)
                {
                    if ( getDoubleSpinBoxValue(toAdd, aIn, aK, ",") ) nbDefVal++;
                }

                if ( getDoubleSpinBoxValue(toAdd, aIn, max, "]") ) nbDefVal++;

                if (nbDefVal < aIn->NbWidgets()) aAdd += toAdd;
            }
            break;
        }
        }

        if (!aAdd.empty())
        {
            if (aIn->IsOpt()) aCom += aIn->Arg().NameArg() + "=" + aAdd + " ";
            else aCom += aAdd + " ";
        }
    }

    if (runCom)
    {
        cout << "VisualMM - Com = " << aCom << endl;
        hide();

        if (_SaisieWin) _SaisieWin->close();

        ::System(aCom);

        if (_showDialog)
        {
            setWindowFlags(Qt::WindowStaysOnTopHint);
            QMessageBox::information(this, QString(argv_recup.c_str()), tr("Job done"));
        }

        //_SaisieWin->close();
        QApplication::exit();
    }
    else
    {
        QMessageBox::critical(this, tr("Error"), tr("Mandatory argument missing!!!"));
    }
}

void visual_MainWindow::onSelectImgsPressed(int aK)
{
    string full_pattern;
    QStringList files = QFileDialog::getOpenFileNames(
                            this,
                            tr("Select images"),
                            mlastDir,
                            tr("Images (*.png *.PNG *.jpg *.JPG *.TIF *.tif *.cr2 *.CR2 *.crw *.CRW *.nef *.NEF);;All Files (*.*)"));

    if (files.size())
    {
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,files[0].toStdString());
        mlastDir = QString(aDir.c_str());

        string fileList="("+ aNameFile;
        for (int i=1;i<files.length();i++)
        {
            SplitDirAndFile(aDir,aNameFile,files[i].toStdString());

            fileList.append("|" + aNameFile);
        }

        fileList+=")";
        full_pattern = QUOTE(aDir+fileList);

        vLineEdit[aK]->setText(QString(full_pattern.c_str()));
        vLineEdit[aK]->setModified(true);
        //cout<<full_pattern.toStdString()<<endl;
    }
}

void visual_MainWindow::onSelectFilePressed(int aK)
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Select file"), mlastDir);

    if (filename != NULL)
    {
        setLastDir(filename);

        vLineEdit[aK]->setText(filename);
        vLineEdit[aK]->setModified(true);
    }
}

void visual_MainWindow::onSelectFileRPPressed(int aK)
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Select file"), mlastDir);

    if (filename != NULL)
    {
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,filename.toStdString());
        mlastDir = QString(aDir.c_str());

        vLineEdit[aK]->setText(QString(aNameFile.c_str()));
        vLineEdit[aK]->setModified(true);
    }
}

void visual_MainWindow::onSelectDirPressed(int aK)
{
    QString aDir = QFileDialog::getExistingDirectory( this, tr("Select directory"), mlastDir);

    if (aDir != NULL)
    {
        QString dirName = QDir(aDir).dirName();

        if (!dirName.isEmpty() && (dirName.right(1) != "/"))
           dirName.append("/");

        mlastDir = "../" + dirName;

        vLineEdit[aK]->setText(dirName);
        vLineEdit[aK]->setModified(true);
    }
}

void visual_MainWindow::setLastDir(QString filename)
{
    string aDir, aNameFile;
    SplitDirAndFile(aDir,aNameFile,filename.toStdString());
    mlastDir = QString(aDir.c_str());
}

void visual_MainWindow::onSaisieButtonPressed(int aK, bool normalize)
{
    QString filename;

    if (_bMaltGeomImg)
    {
        for (unsigned int aK=0;aK<vInputs.size();aK++)
        {
            cInputs* aIn = vInputs[aK];

            if ((aIn->Type() == eIT_LineEdit) && (aIn->Arg().NameArg() == "Master") )
            {
                QLineEdit* lEdit = (QLineEdit*) aIn->Widgets()[0].second;

                filename = mlastDir + lEdit->text().simplified();
            }
        }
    }

    if (filename.isEmpty() || (!QFile(filename).exists()))
         filename = QFileDialog::getOpenFileName(this, tr("Open file"), mlastDir);

    if (filename != NULL)
    {
        setLastDir(filename);
        _curIdx = aK;

        cInputs* aIn = vInputs[aK];

        _m_FileOriMnt.NameFileMnt() = "";

        if(aIn->Type() == eIT_DoubleSpinBox)
        {
            QString nameARG = (aIn->Arg().NameArg().c_str());
            checkGeoref(filename);
            if (_m_FileOriMnt.NameFileMnt() == "" && (nameARG==QString("BoxTerrain")))
            {
                QMessageBox::critical(this, tr("Error"), tr("No xlm file ori mnt"));
                //return;
            }
        }

        _SaisieWin->resize(800,600);
        _SaisieWin->move(200,200);
        _SaisieWin->show();

        QStringList aFiles;
        aFiles.push_back(filename);
        _SaisieWin->addFiles(aFiles);

        _SaisieWin->normalizeCurrentPolygon(normalize);

        connect(_SaisieWin->getWidget(0),SIGNAL(newRectanglePosition(QVector <QPointF>)),this,SLOT(onRectanglePositionChanged(QVector<QPointF>)));

        connect(_SaisieWin,SIGNAL(sgnClose()), this, SLOT(onSaisieQtWindowClosed()));

        if(aIn->Type() == eIT_DoubleSpinBox)
        {
            connect(this,SIGNAL(newX0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newY0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newX1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newY1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(double)));
        }
        else if (aIn->Type() == eIT_SpinBox)
        {
            connect(this,SIGNAL(newX0Position(int)),(QSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newY0Position(int)),(QSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newX1Position(int)),(QSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newY1Position(int)),(QSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(int)));
        }
    }
}

void visual_MainWindow::checkGeoref(QString aNameFile)
{

    QFileInfo fi(aNameFile);
    QString suffix = fi.suffix();
    QString xmlFile = fi.absolutePath() + QDir::separator() + fi.baseName() + ".xml";

    if ((suffix == "tif") && (QFile(xmlFile).exists()))
    {
        std::string aNameTif = aNameFile.toStdString();

        _m_FileOriMnt =  StdGetObjFromFile<cFileOriMnt>
                         (
                             StdPrefix(aNameTif)+".xml",
                             StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                             "FileOriMnt",
                             "FileOriMnt"
                             );
    }

}

const QVector<QPointF> visual_MainWindow::transfoTerrain(QVector<QPointF> res)
{


    if (_m_FileOriMnt.NameFileMnt() != "")
    {
        QVector<QPointF> resTerrain;
        for (int aK=0; aK < res.size(); ++aK)
        {
            Pt2dr ptImage(res[aK].x(), res[aK].y());
            Pt2dr ptTerrain = ToMnt(_m_FileOriMnt, ptImage);
            resTerrain.push_back(QPointF(ptTerrain.x, ptTerrain.y));
        }
        return resTerrain;
    }
    else
        return res;
}

void visual_MainWindow::_adjustSize(int)
{
    adjustSize();
}

void visual_MainWindow::onRectanglePositionChanged(QVector<QPointF> points)
{
    QVector<QPointF> pointsTerrain =  transfoTerrain(points);

    QPointF minPt(1e9,1e9);
    QPointF maxPt(0,0);

    for (int i = 0; i < pointsTerrain.size(); ++i) {

        QPointF pt = pointsTerrain[i];

        if(pt.x()<minPt.x())minPt.setX(pt.x());
        if(pt.y()<minPt.y())minPt.setY(pt.y());
        if(pt.x()>maxPt.x())maxPt.setX(pt.x());
        if(pt.y()>maxPt.y())maxPt.setY(pt.y());
    }

    emit newX0Position((int) minPt.x());
    emit newY0Position((int) minPt.y());
    emit newX1Position((int) maxPt.x());
    emit newY1Position((int) maxPt.y());

    emit newX0Position(minPt.x());
    emit newY0Position(minPt.y());
    emit newX1Position(maxPt.x());
    emit newY1Position(maxPt.y());
}

void visual_MainWindow::onSaisieQtWindowClosed()
{
    cInputs* aIn = vInputs[_curIdx];

    if(aIn->Type() == eIT_DoubleSpinBox)
    {
        disconnect(this,SIGNAL(newX0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newY0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newX1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newY1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(double)));
    }
    else if (aIn->Type() == eIT_SpinBox)
    {
        disconnect(this,SIGNAL(newX0Position(int)),(QSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newY0Position(int)),(QSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newX1Position(int)),(QSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newY1Position(int)),(QSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(int)));
    }
}

void visual_MainWindow::setShowDialog(int state)
{
    if (state == Qt::Checked) _showDialog = true;
    else _showDialog = false;
}

void visual_MainWindow::add_combo(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    list<string> liste_valeur_enum = listPossibleValues(aArg);

    QComboBox* aCombo = new QComboBox(parent);
    layout->addWidget(aCombo,aK,1, 1, 2);

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QComboBox*> (eIT_ComboBox, aCombo));
    vInputs.push_back(new cInputs(aArg, vWidgets));


    list<string>::const_iterator it = liste_valeur_enum.begin();
    for (; it != liste_valeur_enum.end(); it++)
    {
        aCombo->addItem(QString((*it).c_str()));
    }

    if (aArg.Type() == AMBT_bool)
    {
        bool aBool = *(aArg.DefaultValue<bool>());

        if (aBool) aCombo->setCurrentIndex(0);
        else       aCombo->setCurrentIndex(1);
    }
    else if (aArg.Type() == AMBT_string)
    {
        if ( aArg.DefaultValue<string>() != NULL)
        {
            string aStr = *(aArg.DefaultValue<string>());

            if ( aStr.empty() ) aCombo->setCurrentIndex(0);
            else
            {
                int idx = -1;
                int cpt = 0;
                list<string>::const_iterator it = liste_valeur_enum.begin();
                for (; it != liste_valeur_enum.end(); it++, cpt++)
                {
                    if (aStr == *it) idx = cpt;
                }
                if (idx >= 0) aCombo->setCurrentIndex(idx);
            }
        }
    }
    else if (aArg.Type() == AMBT_INT)
    {
        int val = *(aArg.DefaultValue<int>());

        if (val==0) aCombo->setCurrentIndex(1);
        if (val==1) aCombo->setCurrentIndex(0);
        //cout << "other type...." << aArg.NameArg() <<" " << aArg.NameType() <<  " " << val << endl;
    }
}

void visual_MainWindow::add_label(QGridLayout* layout, QWidget* parent, int ak, cMMSpecArg aArg)
{
    QString comment(aArg.Comment().c_str());
    QLabel * com = new QLabel(comment, parent);

    if (aArg.IsOpt())
    {
        com->setText(QString(aArg.NameArg().c_str()));
        com->setToolTip(comment);
    }

    layout->addWidget(com,ak,0);
}

void visual_MainWindow::add_select(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QLineEdit* aLineEdit = new QLineEdit(parent);

    if (aArg.Type() == AMBT_string )
    {
        string val(*(aArg.DefaultValue<string>()));

        if ((val != "") && (val != NoInit)) aLineEdit->setText(QString(val.c_str()));

        if (isFirstArgMalt(val))
        {
            aLineEdit->setEnabled(false);
            aLineEdit->setStyleSheet("QLineEdit{background: lightgrey;}");
        }

        if (aArg.IsPatFile())
        {
            string aDir, aFile;
            SplitDirAndFile(aDir,aFile,val);
            mlastDir = QString(aDir.c_str());
        }
    }

    if (!aArg.IsOutputFile() && !aArg.IsOutputDirOri())
    {
        if (aArg.IsExistDirOri() || aArg.IsDir())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &directory"), (int)vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsPatFile())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &images"), (int)vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsExistFile() )
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &file"), (int)vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsExistFileWithRelativePath())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &file"), (int)vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectFileRPPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
    }

    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,2);

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QLineEdit*> (eIT_LineEdit, aLineEdit));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

QDoubleSpinBox * visual_MainWindow::create_1d_SpinBox(QGridLayout *layout, QWidget *parent, int aK, int bK)
{
    QDoubleSpinBox *aSpinBox = new QDoubleSpinBox(parent);
    layout->addWidget(aSpinBox,aK, bK);

    aSpinBox->setRange(DoubleMin, DoubleMax);

    return aSpinBox;
}

QSpinBox * visual_MainWindow::create_1i_SpinBox(QGridLayout *layout, QWidget *parent, int aK, int bK)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK, bK);

    //aSpinBox->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    // a ne pas utiliser car �a cr�e des spinbox immenses...
    aSpinBox->setRange(IntMin, IntMax);

    return aSpinBox;
}

void visual_MainWindow::add_1d_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *aSpinBox = create_1d_SpinBox(layout, parent, aK, 1);

    aSpinBox->setValue( *(aArg.DefaultValue<double>()) );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, aSpinBox));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2d_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *xSpinBox = create_1d_SpinBox(layout, parent, aK, 1);
    QDoubleSpinBox *ySpinBox = create_1d_SpinBox(layout, parent, aK, 2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3d_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 3;
    for (int i=0; i< nbItems;++i)
    {
        QDoubleSpinBox *spinBox = create_1d_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, spinBox));
    }

    ((QDoubleSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).x );
    ((QDoubleSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).y );
    ((QDoubleSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).z );

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_saisieButton(QGridLayout *layout, int aK, bool normalize)
{
    cSelectionButton *saisieButton = new cSelectionButton(tr("Selection &editor"), (int)vInputs.size(), normalize);
    layout->addWidget(saisieButton, aK, 5);
    connect(saisieButton,SIGNAL(my_click(int, bool)),this,SLOT(onSaisieButtonPressed(int, bool)));
}

void visual_MainWindow::add_4d_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 4;
    for (int i=0; i< nbItems;++i)
    {
        QDoubleSpinBox *spinBox = create_1d_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, spinBox));
    }

    ((QDoubleSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).x(0) );
    ((QDoubleSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).y(0) );
    ((QDoubleSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).x(1) );
    ((QDoubleSpinBox*)(vWidgets[3].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).y(1) );

    add_saisieButton(layout, aK, aArg.IsToNormalize());

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_1i_SpinBox(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    if (aArg.IsPowerOf2())
    {
        cSpinBox *aSpinBox = new cSpinBox(*(aArg.DefaultValue<int>()), parent);
        layout->addWidget(aSpinBox,aK, 1);

        vWidgets.push_back(pair <int, cSpinBox*> (eIT_SpinBox, aSpinBox));
    }
    else
    {
        QSpinBox *aSpinBox = create_1i_SpinBox(layout, parent, aK, 1);

        aSpinBox->setValue( *(aArg.DefaultValue<int>()) );
        vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, aSpinBox));
    }

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2i_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *xSpinBox = create_1i_SpinBox(layout, parent, aK, 1);
    QSpinBox *ySpinBox = create_1i_SpinBox(layout, parent, aK, 2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3i_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 3;
    for (int i=0; i< nbItems;++i)
    {
        QSpinBox *spinBox = create_1i_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, spinBox));
    }

    ((QSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).x );
    ((QSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).y );
    ((QSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).z );

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_4i_SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 4;
    for (int i=0; i< nbItems;++i)
    {
        QSpinBox *spinBox = create_1i_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, spinBox));
    }

    ((QSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Box2di>())).x(0) );
    ((QSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Box2di>())).y(0) );
    ((QSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Box2di>())).x(1) );
    ((QSpinBox*)(vWidgets[3].second))->setValue( (*(aArg.DefaultValue<Box2di>())).y(1) );

    add_saisieButton(layout, aK, aArg.IsToNormalize());

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;

    setWindowTitle( QString((argv + " " + mFirstArg).c_str()) );
}

void visual_MainWindow::resizeEvent(QResizeEvent *)
{
    QDesktopWidget* m = qApp->desktop();
    QRect desk_rect = m->screenGeometry(m->screenNumber(QCursor::pos()));

    int desk_x = desk_rect.width();
    int desk_y = desk_rect.height();

    move(desk_x / 2 - width() / 2 + desk_rect.left(), desk_y / 2 - height() / 2 + desk_rect.top());
}

void visual_MainWindow::closeEvent(QCloseEvent *)
{
    saveSettings();
}

void visual_MainWindow::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_Enter || event->key() == Qt::Key_Return)
    {
        onRunCommandPressed();
    }
}

cInputs::cInputs(cMMSpecArg aArg, vector<pair<int, QWidget *> > aWid):
    mArg(aArg),
    vWidgets(aWid)
{

}

int cInputs::Type()
{
    if (vWidgets.size()) return vWidgets[0].first;  //todo: verifier que les arguments multiples sont tous du m�me type....
    else return eIT_None;
}

#endif // ELISE_QT



#include "general/visual_mainwindow.h"

#if (ELISE_QT_VERSION >= 4)

// aVAM: Mandatory args
// aVAO: Optional args

static int IntMin = -1000000;
static int IntMax =  1000000;
static double DoubleMin = -1000000.;
static double DoubleMax =  1000000.;

visual_MainWindow::visual_MainWindow(vector<cMMSpecArg> & aVAM,
                                     vector<cMMSpecArg> & aVAO,
                                     string aFirstArg,
                                     QWidget *parent):
    QWidget(parent),
    mlastDir(QDir::currentPath()),
    mFirstArg(aFirstArg)
{
    moveArgs(aVAM, aVAO);

    QVBoxLayout *verticalLayout = new QVBoxLayout(this);

    setLayout(verticalLayout);

    toolBox = new QToolBox();

    verticalLayout->addWidget(toolBox);

    addGridLayout(aVAM, tr("&Mandatory arguments"));

    if (aVAO.size())
    {
        addGridLayout(aVAO, tr("&Optional arguments"));

        connect(toolBox, SIGNAL(currentChanged(int)), this, SLOT(_adjustSize(int)));
    }

    runCommandButton = new QPushButton(tr(" &Run command "), this);

    verticalLayout->addWidget(runCommandButton, 1, Qt::AlignRight);

    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));
}

visual_MainWindow::~visual_MainWindow()
{
    delete toolBox;
    delete runCommandButton;
}

void visual_MainWindow::moveArgs(vector<cMMSpecArg> &aVAM, vector<cMMSpecArg> &aVAO)
{
    bool bGeomImg = false;

    for (int aK=0; aK < (int) aVAM.size(); aK++)
    {
        cMMSpecArg arg = aVAM[aK];

        if (( arg.Type() == AMBT_string ) && (*(arg.DefaultValue<string>()) != "GeomImage")) bGeomImg = true;
        if (bGeomImg) break;
    }

    if (bGeomImg)
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

        for (int aK= idx; aK < (int) aVAO.size() -1; aK++)
            aVAO[aK] = aVAO[aK+1];
        aVAO.pop_back();
    }
}

void visual_MainWindow::addGridLayout(const vector<cMMSpecArg> &aVA, QString pageName)
{
    QWidget* mPage = new QWidget();

    toolBox->addItem(mPage, pageName);

    QGridLayout* gridLayout = new QGridLayout(mPage);

    buildUI(aVA, gridLayout, mPage);
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
                add_4SpinBox(layout, parent, aK, aArg);
                break;
            case AMBT_Box2dr:
                add_4dSpinBox(layout, parent, aK, aArg);
                break;
            case AMBT_bool:
                add_combo(layout, parent, aK, aArg);
                break;
            case AMBT_INT:
            case AMBT_U_INT1:
            case AMBT_INT1:
                add_spinBox(layout, parent, aK, aArg);
                break;
            case AMBT_REAL:
                add_dSpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt2di:
                add_2SpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt2dr:
                add_2dSpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt3dr:
                add_3dSpinBox(layout, parent, aK, aArg);
            break;
            case AMBT_Pt3di:
                add_3SpinBox(layout, parent, aK, aArg);
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

    QSpacerItem *vSpacer = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

    layout->addItem(vSpacer, aVA.size(), 0);
}

bool visual_MainWindow::getSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar)
{
    int val = ((QSpinBox*) aIn->Widgets()[aK].second)->value();

    stringstream ss;
    ss << val;
    aAdd += ss.str() + endingCar;

    if ( aIn->Arg().IsDefaultValue<int>(val) )
        return true;
    else
        return false;
}

bool visual_MainWindow::getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar)
{
    double val = ((QDoubleSpinBox*) aIn->Widgets()[aK].second)->value();

    stringstream ss;
    ss << val;
    aAdd += ss.str() + endingCar;

    if ( aIn->Arg().IsDefaultValue<double>(val) )
        return true;
    else
        return false;
}

void visual_MainWindow::onRunCommandPressed()
{
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

                if (lEdit->isModified())
                {
                    string aStr = lEdit->text().toStdString();

                    if ( !aStr.empty() ) aAdd += aStr;
                    else if (!aIn->IsOpt()) runCom = false;
                }
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
            if (aIn->IsOpt()) aCom += aIn->Arg().NameArg()+ "=" + aAdd + " ";
            else aCom += aAdd + " ";
        }
    }

    if (runCom)
    {
        cout << "VisualMM - Com = " << aCom << endl;
        hide();

        ::System(aCom);

        QMessageBox::information(this, QString(argv_recup.c_str()), tr("Job finished"));
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
                            tr("Images (*.png *.jpg *.tif *.cr2 *.crw *.nef);;Images (*.*)"));

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
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,filename.toStdString());
        mlastDir = QString(aDir.c_str());

        vLineEdit[aK]->setText(filename);
        vLineEdit[aK]->setModified(true);
    }
}

void visual_MainWindow::onSelectDirPressed(int aK)
{
    QString aDir = QFileDialog::getExistingDirectory( this, tr("Select directory"), mlastDir);

    if (aDir != NULL)
    {
        mlastDir = "../" + aDir;

        vLineEdit[aK]->setText(QDir(aDir).dirName());
        vLineEdit[aK]->setModified(true);
    }
}

void visual_MainWindow::onSaisieButtonPressed(int aK, bool normalize)
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open file"), mlastDir);

    if (filename != NULL)
    {
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,filename.toStdString());
        mlastDir = QString(aDir.c_str());

        _SaisieWin->resize(800,600);
        _SaisieWin->move(200,200);
        _SaisieWin->show();

        QStringList aFiles;
        aFiles.push_back(filename);
        _SaisieWin->addFiles(aFiles);

        _SaisieWin->normalizeCurrentPolygon(normalize);

        connect(_SaisieWin->getWidget(0),SIGNAL(newRectanglePosition(QVector <QPointF>)),this,SLOT(onRectanglePositionChanged(QVector<QPointF>)));

        connect(_SaisieWin,SIGNAL(sgnClose()), this, SLOT(onSaisieQtWindowClosed()));

        _curIdx = aK;

        cInputs* aIn = vInputs[aK];

        if(aIn->Type() == eIT_DoubleSpinBox)
        {
            connect(this,SIGNAL(newX0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newX1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newY0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(double)));
            connect(this,SIGNAL(newY1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(double)));
        }
        else if (aIn->Type() == eIT_SpinBox)
        {
            connect(this,SIGNAL(newX0Position(int)),(QSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newX1Position(int)),(QSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newY0Position(int)),(QSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(int)));
            connect(this,SIGNAL(newY1Position(int)),(QSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(int)));
        }
    }
}

void visual_MainWindow::_adjustSize(int)
{
    adjustSize();
}

void visual_MainWindow::onRectanglePositionChanged(QVector<QPointF> points)
{
    emit newX0Position((int) points[0].x());
    emit newY0Position((int) points[0].y());
    emit newX1Position((int) points[2].x());
    emit newY1Position((int) points[2].y());

    emit newX0Position(points[0].x());
    emit newY0Position(points[0].y());
    emit newX1Position(points[2].x());
    emit newY1Position(points[2].y());
}

void visual_MainWindow::onSaisieQtWindowClosed()
{
    cInputs* aIn = vInputs[_curIdx];

    if(aIn->Type() == eIT_DoubleSpinBox)
    {
        disconnect(this,SIGNAL(newX0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newX1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newY0Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(double)));
        disconnect(this,SIGNAL(newY1Position(double)),(QDoubleSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(double)));
    }
    else if (aIn->Type() == eIT_SpinBox)
    {
        disconnect(this,SIGNAL(newX0Position(int)),(QSpinBox*)(aIn->Widgets()[0].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newX1Position(int)),(QSpinBox*)(aIn->Widgets()[1].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newY0Position(int)),(QSpinBox*)(aIn->Widgets()[2].second), SLOT(setValue(int)));
        disconnect(this,SIGNAL(newY1Position(int)),(QSpinBox*)(aIn->Widgets()[3].second), SLOT(setValue(int)));
    }
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
        string defVal(*(aArg.DefaultValue<string>()));
        if (defVal != "") aLineEdit->setText(QString(defVal.c_str()));
    }

    if (!aArg.IsOutputFile() && !aArg.IsOutputDirOri())
    {
        if (aArg.IsExistDirOri() || aArg.IsDir())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &directory"), vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsPatFile())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &images"), vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsExistFile())
        {
            cSelectionButton* sButton = new cSelectionButton(tr("Select &file"), vLineEdit.size(), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
            layout->addWidget(sButton,aK,3);
        }
    }

    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,2);

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QLineEdit*> (eIT_LineEdit, aLineEdit));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

QDoubleSpinBox * visual_MainWindow::create_dSpinBox(QGridLayout *layout, QWidget *parent, int aK, int bK)
{
    QDoubleSpinBox *aSpinBox = new QDoubleSpinBox(parent);
    layout->addWidget(aSpinBox,aK, bK);

    aSpinBox->setRange(DoubleMin, DoubleMax);

    return aSpinBox;
}

QSpinBox * visual_MainWindow::create_SpinBox(QGridLayout *layout, QWidget *parent, int aK, int bK)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK, bK);

    //aSpinBox->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    // a ne pas utiliser car ça crée des spinbox immenses...
    aSpinBox->setRange(IntMin, IntMax);

    return aSpinBox;
}

void visual_MainWindow::add_dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *aSpinBox = create_dSpinBox(layout, parent, aK, 1);

    aSpinBox->setValue( *(aArg.DefaultValue<double>()) );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, aSpinBox));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *xSpinBox = create_dSpinBox(layout, parent, aK, 1);
    QDoubleSpinBox *ySpinBox = create_dSpinBox(layout, parent, aK, 2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 3;
    for (int i=0; i< nbItems;++i)
    {
        QDoubleSpinBox *spinBox = create_dSpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, spinBox));
    }

    ((QDoubleSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).x );
    ((QDoubleSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).y );
    ((QDoubleSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Pt3dr>())).z );

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_saisieButton(QGridLayout *layout, int aK, bool normalize)
{
    cSelectionButton *saisieButton = new cSelectionButton(tr("Selection &editor"), vInputs.size(), normalize);
    layout->addWidget(saisieButton, aK, 5);
    connect(saisieButton,SIGNAL(my_click(int, bool)),this,SLOT(onSaisieButtonPressed(int, bool)));
}

void visual_MainWindow::add_4dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 4;
    for (int i=0; i< nbItems;++i)
    {
        QDoubleSpinBox *spinBox = create_dSpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, spinBox));
    }

    ((QDoubleSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).x(0) );
    ((QDoubleSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).x(1) );
    ((QDoubleSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).y(0) );
    ((QDoubleSpinBox*)(vWidgets[3].second))->setValue( (*(aArg.DefaultValue<Box2dr>())).y(1) );

    add_saisieButton(layout, aK, aArg.IsToNormalize());

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_spinBox(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *aSpinBox = create_SpinBox(layout, parent, aK, 1);

    aSpinBox->setValue( *(aArg.DefaultValue<int>()) );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, aSpinBox));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *xSpinBox = create_SpinBox(layout, parent, aK, 1);
    QSpinBox *ySpinBox = create_SpinBox(layout, parent, aK, 2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 4;
    for (int i=0; i< nbItems;++i)
    {
        QSpinBox *spinBox = create_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, spinBox));
    }

    ((QSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).x );
    ((QSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).y );
    ((QSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Pt3di>())).z );

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_4SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    vector< pair < int, QWidget * > > vWidgets;

    int nbItems = 4;
    for (int i=0; i< nbItems;++i)
    {
        QSpinBox *spinBox = create_SpinBox(layout, parent, aK, i+1);

        vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, spinBox));
    }

    ((QSpinBox*)(vWidgets[0].second))->setValue( (*(aArg.DefaultValue<Box2di>())).x(0) );
    ((QSpinBox*)(vWidgets[1].second))->setValue( (*(aArg.DefaultValue<Box2di>())).x(1) );
    ((QSpinBox*)(vWidgets[2].second))->setValue( (*(aArg.DefaultValue<Box2di>())).y(0) );
    ((QSpinBox*)(vWidgets[3].second))->setValue( (*(aArg.DefaultValue<Box2di>())).y(1) );

    add_saisieButton(layout, aK, aArg.IsToNormalize());

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;

    if (mFirstArg != "") setWindowTitle( QString((argv + " " + mFirstArg).c_str()) );
}

void visual_MainWindow::resizeEvent(QResizeEvent *)
{
    //deplacement au centre de l'ecran
    const QPoint global = qApp->desktop()->availableGeometry().center();
    move(global.x() - width() / 2, global.y() - height() / 2);
}

cInputs::cInputs(cMMSpecArg aArg, vector<pair<int, QWidget *> > aWid):
    mArg(aArg),
    vWidgets(aWid)
{

}

int cInputs::Type()
{
    if (vWidgets.size()) return vWidgets[0].first;  //todo: verifier que les arguments multiples sont tous du même type....
    else return eIT_None;
}

#endif //ELISE_QT_VERSION >= 4



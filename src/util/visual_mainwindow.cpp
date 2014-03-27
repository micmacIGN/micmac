#if(ELISE_QT5)

#include "general/visual_mainwindow.h"
#include "general/visual_buttons.h"

#include "StdAfx.h"

// aVAM: Mandatory args
// aVAO: Optional args

static int IntMin = -1000;
static int IntMax =  1000;
static double DoubleMin = -10000.;
static double DoubleMax =  10000.;

visual_MainWindow::visual_MainWindow(vector<cMMSpecArg> & aVAM, vector<cMMSpecArg> & aVAO, QWidget *parent) :
    QMainWindow(parent),
    mlastDir(QDir::homePath())
{
    mainWidget = new QWidget(this);
    setCentralWidget(mainWidget);

    QVBoxLayout *verticalLayout = new QVBoxLayout(mainWidget);

    mainWidget->setLayout(verticalLayout);

    toolBox = new QToolBox();

    verticalLayout->addWidget(toolBox);

    addGridLayout(aVAM, tr("Mandatory arguments"));

    if (aVAO.size())
    {
        addGridLayout(aVAO, tr("Optional arguments"));

        connect(toolBox, SIGNAL(currentChanged(int)), this, SLOT(_adjustSize(int)));
    }

    runCommandButton = new QPushButton(tr(" Run command "), mainWidget);

    verticalLayout->addWidget(runCommandButton, 1, Qt::AlignRight);

    connect(runCommandButton,SIGNAL(clicked()),this,SLOT(onRunCommandPressed()));
}

visual_MainWindow::~visual_MainWindow()
{
    delete mainWidget;
    delete toolBox;
    delete runCommandButton;
}

void visual_MainWindow::addGridLayout(vector<cMMSpecArg>& aVA, QString pageName)
{
    QWidget* mPage = new QWidget();

    toolBox->addItem(mPage, pageName);

    QGridLayout* gridLayout = new QGridLayout(mPage);

    buildUI(aVA, gridLayout, mPage);
}

void visual_MainWindow::buildUI(vector<cMMSpecArg>& aVA, QGridLayout *layout, QWidget *parent)
{
    for (int aK=0 ; aK<int(aVA.size()) ; aK++)
    {
        cMMSpecArg aArg = aVA[aK];
        //cout << "arg " << aK << " ; Type is " << aArg.NameType() << " ; Name is " << aArg.NameArg() <<  endl;

        add_label(layout, parent, aK, aArg);

        if (aArg.IsBool()) // because some boolean values are set with int
        {
            add_combo(layout, parent, aK, aArg);
        }
        else
        {
            switch (aArg.Type())
            {
            case AMBT_Box2di:
                break;
            case AMBT_Box2dr:
                break;
            case AMBT_bool:
                add_combo(layout, parent, aK, aArg);
                break;
            case AMBT_INT:
            case AMBT_U_INT1:
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
                {
                    add_combo(layout, parent, aK, aArg);
                }
                else
                {
                    add_select(layout, parent, aK, aArg);
                }
                break;
            }
            case AMBT_INT1:
            case AMBT_char:
            break;
            case AMBT_vector_Pt2dr:
            break;
            case AMBT_vector_int:
            break;
            case AMBT_vector_double:
            break;
            case AMBT_vvector_int:
            break;
            case AMBT_vector_string:
                add_select(layout, parent, aK, aArg);
            break;
            case AMBT_unknown:
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

    if ( aIn->Arg().IsDefaultValue<int>(val) )
    {
        return true;
    }
    else
    {
        stringstream ss;
        ss << val;
        aAdd += ss.str() + endingCar;

        return false;
    }
}

bool visual_MainWindow::getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar)
{
    double val = ((QDoubleSpinBox*) aIn->Widgets()[aK].second)->value();

    if ( aIn->Arg().IsDefaultValue<double>(val) )
    {
        return true;
    }
    else
    {
        stringstream ss;
        ss << val;
        aAdd += ss.str() + endingCar;

        return false;
    }
}

void visual_MainWindow::onRunCommandPressed()
{
    bool runCom = true;

    string aCom = MM3dBinFile(argv_recup) +" ";

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
                    string aStr = ((QLineEdit*) aIn->Widgets()[0].second)->text().toStdString();
                    if (!aStr.empty()) aAdd += aStr;
                    else if (!aIn->IsOpt()) runCom = false;
                }
                break;
            }
            case eIT_ComboBox:
            {
                if (aIn->Widgets().size() == 1)
                {
                    QString aStr = ((QComboBox*) aIn->Widgets()[0].second)->currentText(); //warning

                    if(aIn->Arg().IsBool() )
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
                    else
                        aAdd += aStr.toStdString();
                }
                break;
            }
            case eIT_SpinBox:
            {
                if (aIn->Widgets().size() == 1)
                {
                    getSpinBoxValue(aAdd, aIn, 0);
                }
                else
                {
                    string toAdd = "[";
                    int nbDefVal = 0;

                    int max = aIn->NbWidgets()-1;

                    for (int aK=0; aK < max;++aK)
                    {
                        if ( getSpinBoxValue(toAdd, aIn, aK, ";") ) nbDefVal++;
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
                    getDoubleSpinBoxValue(aAdd, aIn, 0);
                }
                else
                {
                    string toAdd = "[";
                    int nbDefVal = 0;

                    int max = aIn->NbWidgets()-1;

                    for (int aK=0; aK < max ;++aK)
                    {
                        if ( getDoubleSpinBoxValue(toAdd, aIn, aK, ";") ) nbDefVal++;
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
        cout << "Com = " << aCom << endl;
        int aRes = ::System(aCom);

        cout << "----------------- " << aRes << endl;
    }
    else
    {
        QMessageBox::critical(this, tr("Error"), tr("Mandatory argument not filled!!!"));
    }
}

void visual_MainWindow::onSelectImgsPressed(int aK)
{
    string full_pattern;
    QStringList files = QFileDialog::getOpenFileNames(
                            mainWidget,
                            tr("Select images"),
                            mlastDir,
                            tr("Images (*.png *.xpm *.jpg *.tif)"));

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
        //cout<<full_pattern.toStdString()<<endl;

        adjustSize();
    }
}

void visual_MainWindow::onSelectFilePressed(int aK)
{
    QString filename = QFileDialog::getOpenFileName(mainWidget, tr("Select file"), mlastDir);

    if (filename != NULL)
    {
        string aDir, aNameFile;
        SplitDirAndFile(aDir,aNameFile,filename.toStdString());
        mlastDir = QString(aDir.c_str());

        vLineEdit[aK]->setText(filename);

        adjustSize();
    }
}

void visual_MainWindow::onSelectDirPressed(int aK)
{
    QString aDir = QFileDialog::getExistingDirectory( mainWidget, tr("Select directory"), mlastDir);

    if (aDir != NULL)
    {
        mlastDir = aDir;

        vLineEdit[aK]->setText(QDir(aDir).dirName());

        adjustSize();
    }
}

void visual_MainWindow::_adjustSize(int)
{
    adjustSize();
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

            if ( aStr.empty() )  aCombo->setCurrentIndex(0);
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
    vLineEdit.push_back(aLineEdit);
    layout->addWidget(aLineEdit,aK,1,1,2);

    if (!aArg.IsOutputFile() && !aArg.IsOutputDirOri())
    {
        if (aArg.IsExistDirOri())
        {
            selectionButton* sButton = new selectionButton(tr("Select directory"), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectDirPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsPatFile())
        {
            selectionButton* sButton = new selectionButton(tr("Select images"), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectImgsPressed(int)));
            layout->addWidget(sButton,aK,3);
        }
        else if (aArg.IsExistFile())
        {
            selectionButton* sButton = new selectionButton(tr("Select file"), parent);
            connect(sButton,SIGNAL(my_click(int)),this,SLOT(onSelectFilePressed(int)));
            layout->addWidget(sButton,aK,3);
        }
    }

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QLineEdit*> (eIT_LineEdit, aLineEdit));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *aSpinBox = new QDoubleSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1);

    aSpinBox->setRange(DoubleMin, DoubleMax);
    aSpinBox->setValue( *(aArg.DefaultValue<double>()) );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, aSpinBox));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *xSpinBox = new QDoubleSpinBox(parent);
    QDoubleSpinBox *ySpinBox = new QDoubleSpinBox(parent);

    xSpinBox->setRange(DoubleMin, DoubleMax);
    ySpinBox->setRange(DoubleMin, DoubleMax);

    layout->addWidget(xSpinBox,aK,1);
    layout->addWidget(ySpinBox,aK,2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2dr>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3dSpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QDoubleSpinBox *xSpinBox = new QDoubleSpinBox(parent);
    QDoubleSpinBox *ySpinBox = new QDoubleSpinBox(parent);
    QDoubleSpinBox *zSpinBox = new QDoubleSpinBox(parent);

    xSpinBox->setRange(DoubleMin, DoubleMax);
    ySpinBox->setRange(DoubleMin, DoubleMax);
    zSpinBox->setRange(DoubleMin, DoubleMax);

    layout->addWidget(xSpinBox,aK,1);
    layout->addWidget(ySpinBox,aK,2);
    layout->addWidget(zSpinBox,aK,3);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt3dr>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt3dr>())).y );
    zSpinBox->setValue( (*(aArg.DefaultValue<Pt3dr>())).z );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, ySpinBox));
    vWidgets.push_back(pair <int, QDoubleSpinBox*> (eIT_DoubleSpinBox, zSpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_spinBox(QGridLayout* layout, QWidget* parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *aSpinBox = new QSpinBox(parent);
    layout->addWidget(aSpinBox,aK,1);

    //aSpinBox->setRange(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    aSpinBox->setRange(IntMin, IntMax);

    aSpinBox->setValue( *(aArg.DefaultValue<int>()) );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, aSpinBox));
    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_2SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *xSpinBox = new QSpinBox(parent);
    QSpinBox *ySpinBox = new QSpinBox(parent);

    xSpinBox->setRange(IntMin, IntMax);
    ySpinBox->setRange(IntMin, IntMax);

    layout->addWidget(xSpinBox,aK,1);
    layout->addWidget(ySpinBox,aK,2);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt2di>())).y );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, ySpinBox));

    vInputs.push_back(new cInputs(aArg, vWidgets));
}

void visual_MainWindow::add_3SpinBox(QGridLayout *layout, QWidget *parent, int aK, cMMSpecArg aArg)
{
    QSpinBox *xSpinBox = new QSpinBox(parent);
    QSpinBox *ySpinBox = new QSpinBox(parent);
    QSpinBox *zSpinBox = new QSpinBox(parent);

    xSpinBox->setRange(IntMin, IntMax);
    ySpinBox->setRange(IntMin, IntMax);
    zSpinBox->setRange(IntMin, IntMax);

    layout->addWidget(xSpinBox,aK,1);
    layout->addWidget(ySpinBox,aK,2);
    layout->addWidget(zSpinBox,aK,3);

    xSpinBox->setValue( (*(aArg.DefaultValue<Pt3di>())).x );
    ySpinBox->setValue( (*(aArg.DefaultValue<Pt3di>())).y );
    zSpinBox->setValue( (*(aArg.DefaultValue<Pt3di>())).z );

    vector< pair < int, QWidget * > > vWidgets;
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, xSpinBox));
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, ySpinBox));
    vWidgets.push_back(pair <int, QSpinBox*> (eIT_SpinBox, zSpinBox));
}

void visual_MainWindow::set_argv_recup(string argv)
{
    argv_recup = argv;
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

#endif //ELISE_QT5







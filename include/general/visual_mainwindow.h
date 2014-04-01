#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#if(ELISE_QT5)



#include "StdAfx.h"

#ifdef Int
    #undef Int
#endif

#include <QMainWindow>

#include <QGridLayout>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QToolBox>
#include <QFileDialog>
#include <QDesktopWidget>
#include <QApplication>
#include <QMessageBox>

using namespace std;

enum eInputType
{
    eIT_LineEdit,
    eIT_ComboBox,
    eIT_SpinBox,
    eIT_DoubleSpinBox,
    eIT_None
};

class cInputs
{
public:
    cInputs(cMMSpecArg, vector < pair < int, QWidget* > >);

    bool        IsOpt()     { return mArg.IsOpt(); }
    cMMSpecArg  Arg()       { return mArg;    }
    int         Type();
    vector < pair < int, QWidget*> >    Widgets()    { return vWidgets; }

private:
    cMMSpecArg  mArg;
    vector < pair < int, QWidget* > >   vWidgets;
};

class visual_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    visual_MainWindow(vector<cMMSpecArg> & aVAM, vector<cMMSpecArg> & aVAO, QWidget *parent = 0);
    ~visual_MainWindow();

    void add_combo     (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_comment   (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_select    (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_spinBox   (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2SpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3SpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_dSpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2dSpinBox (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3dSpinBox (QGridLayout*, QWidget*, int, cMMSpecArg);

    void set_argv_recup(string);

    void buildUI(vector<cMMSpecArg>& aVA, QGridLayout* layout, QWidget* parent);

    void addGridLayout(vector<cMMSpecArg>& aVA, QString pageName);

    void getSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");
    void getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");

public slots:

    void onRunCommandPressed();
    void onSelectFilePressed(int);
    void onSelectImgsPressed(int);
    void onSelectDirPressed(int);
    void _adjustSize(int);

protected:

    void resizeEvent(QResizeEvent *);

    int          id_unique;
    string       argv_recup;

    QWidget*     mainWidget;

    QToolBox*    toolBox;
    QPushButton* runCommandButton;

    vector <QLineEdit*> vLineEdit;      //LineEdit: display what has been selected (images, files, directories)

    vector <cInputs*>   vInputs;

    QString             mlastDir;
};

list<string> listPossibleValues(const cMMSpecArg & anArg);
void ShowEnum(const cMMSpecArg & anArg);

#endif //ELISE_QT5

#endif // MAINWINDOW_H

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "StdAfx.h"

#if (ELISE_QT_VERSION >= 4)

#ifdef Int
    #undef Int
#endif

#include <QWidget>
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

#include "general/visual_buttons.h"

#include "../../src/saisieQT/saisieQT_window.h"

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
    int         NbWidgets() { return vWidgets.size(); }

private:
    cMMSpecArg  mArg;
    vector < pair < int, QWidget* > >   vWidgets;
};

class visual_MainWindow : public QWidget
{
    Q_OBJECT

public:
    visual_MainWindow(const vector<cMMSpecArg> & aVAM,
                      const vector<cMMSpecArg> & aVAO,
                      std::string aFirstArg = "",
                      QWidget *parent = 0);

    ~visual_MainWindow();

    void add_label     (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_combo     (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_select    (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_spinBox   (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2SpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3SpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_4SpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_dSpinBox  (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_2dSpinBox (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_3dSpinBox (QGridLayout*, QWidget*, int, cMMSpecArg);
    void add_4dSpinBox (QGridLayout*, QWidget*, int, cMMSpecArg);

    void set_argv_recup(string);

    void buildUI(const vector<cMMSpecArg>& aVA, QGridLayout* layout, QWidget* parent);

    void addGridLayout(const vector<cMMSpecArg>& aVA, QString pageName);

    bool getSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");
    bool getDoubleSpinBoxValue(string &aAdd, cInputs* aIn, int aK, string endingCar ="");

    QDoubleSpinBox* create_dSpinBox(QGridLayout*, QWidget*, int, int);
    QSpinBox *create_SpinBox(QGridLayout*, QWidget*, int, int);

    std::string getFirstArg() { return mFirstArg; }

    void add_saisieButton(/*vector< pair < int, QWidget * > > vWidgets,*/ QGridLayout *layout, int aK);

    void setSaisieWin(SaisieQtWindow* win){ _SaisieWin = win;}

public slots:

    void onRunCommandPressed();
    void onSelectFilePressed(int);
    void onSelectImgsPressed(int);
    void onSelectDirPressed(int);
    void onSaisieButtonPressed(int);
    void _adjustSize(int);

    void onRectanglePositionChanged(QVector <QPointF>);

signals:

    void clickedInLine(int);

protected:

    void resizeEvent(QResizeEvent *);

    int          id_unique;
    string       argv_recup;

    QToolBox*    toolBox;
    QPushButton* runCommandButton;

    vector <QLineEdit*> vLineEdit;    //LineEdit: display what has been selected (images, files, directories)

    vector <cInputs*>   vInputs;

    QString             mlastDir;

    string              mFirstArg;    //truc&astuces: stores the first arg (for Tapioca)

    SaisieQtWindow*     _SaisieWin;
};

list<string> listPossibleValues(const cMMSpecArg & anArg);
void ShowEnum(const cMMSpecArg & anArg);
void setStyleSheet(QApplication &app);
void showErrorMsg(QApplication &app, std::vector <std::string> vStr);

#endif //ELISE_QT_VERSION >= 4

#endif // MAINWINDOW_H

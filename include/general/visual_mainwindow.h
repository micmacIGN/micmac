#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#if(ELISE_QT5)

#include <QMainWindow>

#include <QGridLayout>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QFileDialog>
#include <QDesktopWidget>
#include <QApplication>

using namespace std;

enum eInputType
{
    lineedit,
    combobox,
    integer
};

class visual_MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    visual_MainWindow(QWidget *parent = 0);
    ~visual_MainWindow();
    void add_combo_line(QString);
    void create_combo(int, std::list<std::string>);
    void create_select_images(int);
    void create_select_orientation(int);
    void create_comment(std::string, int);
    void create_selectFile_button(int);
    void create_champ_int(int);
    void set_argv_recup(std::string);

protected:

    void resizeEvent(QResizeEvent *);

    int id_unique;
    std::string argv_recup;

    QWidget *       gridLayoutWidget;//il faut forcement passer par un QWidget pour le mettre en "CentralWidget" de la MainWindow
    QGridLayout *   gridLayout;
    QLabel *label;
    QComboBox *Combo;
    QComboBox *Combo2;
    QLineEdit *selectFile_LineEdit;
    QPushButton *selectFile_Button;
    QSpinBox *Spin;

    vector <QComboBox*> vEnumValues;    //valeur enumerees
    vector <QLineEdit*> vImageFiles;    //fichiers image
    vector <QLabel*> vCommentaries;     //commentaires
    QPushButton *runCommandButton;
    QString commande;
    vector <eInputType> inputTypes;
    vector <QWidget*> inputs;

public slots:
    void onRunCommandPressed();
    void onSelectFilePressed(int);

};

#endif //ELISE_QT5

#endif // MAINWINDOW_H

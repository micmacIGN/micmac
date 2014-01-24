


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


enum Type_input
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
    void ajoute_ligne_combo(QString);
    void create_combo(int, std::list<std::string>);
    void create_select_images(int);
    void create_select_orientation(int);
    void create_comment(std::string, int);
    void create_bouton_parcourir(int);
    void create_champ_int(int);
    void set_argv_recup(std::string);

protected:

    int id_unique;
    std::string argv_recup;

    QWidget *gridLayoutWidget;//il faut forcement passer par un QWidget pour le mettre en "CentralWidget" de la MainWindow
    QGridLayout *gridLayout;
    QLabel *label;
    QComboBox *Combo;
    QComboBox *Combo2;
    QLineEdit *Sel_Fichiers;
    QPushButton *Parcourir;
    QSpinBox *Spin;
    std::vector <QComboBox*> vecteur_val_enumerees;
    std::vector <QLineEdit*> vecteur_Fichiers_images;
    std::vector <QLabel*> vecteur_Commentaires;
    QPushButton *Valider;
    QString commande;
    std::vector<Type_input> types_inputs;
    std::vector<QWidget*> inputs;

public slots:
    void press_valider();
    void press_parcours(int);

};

#endif //ELISE_QT5

#endif // MAINWINDOW_H

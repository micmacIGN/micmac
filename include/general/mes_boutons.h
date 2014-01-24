#ifndef mes_boutons_H
#define mes_boutons_H

#include <QPushButton>

class mon_bouton_parcours: public QPushButton
{
    Q_OBJECT

    public:
        mon_bouton_parcours(QWidget * parent=0);
        virtual ~mon_bouton_parcours();
    protected:
        int id_unique;
        static int nbr_boutons_exist;

    public slots:
        void onClick();

    signals:
        void mon_click(int);
};

#endif /* MonPushButton_H */

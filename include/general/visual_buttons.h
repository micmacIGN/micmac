#ifndef VISUAL_BUTTONS_H
#define VISUAL_BUTTONS_H

#if(ELISE_QT5)

#include <QPushButton>

class selectionButton: public QPushButton
{
    Q_OBJECT

public:
        selectionButton(QWidget *parent=0);
        virtual ~selectionButton();

        QAction *Button;

protected:
        int unique_id;
        static int buttonNumber;

public slots:
        void onClick();

signals:
        void my_click(int);
};

#endif //ELISE_QT5

#endif /* VISUAL_BUTTONS_H */

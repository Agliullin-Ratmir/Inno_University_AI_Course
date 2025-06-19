###  Задание 0

-- Создание таблицы students
     CREATE TABLE students (
     id SERIAL PRIMARY KEY,
     name TEXT NOT NULL,
     total_score INTEGER DEFAULT 0,
     scholarship INTEGER DEFAULT 0
     );

-- Создание таблицы activity_scores
CREATE TABLE activity_scores (
student_id INTEGER NOT NULL,
activity_type TEXT NOT NULL,
score INTEGER NOT NULL,
FOREIGN KEY (student_id)
REFERENCES students(id)
ON DELETE CASCADE
);

###  Задание 1

1) INSERT INTO students (name)
VALUES
('Андрей Аршавин'),
('Александр Кержаков'),
('Игорь Акинфеев');

2) INSERT INTO activity_scores (student_id, activity_type, score) VALUES
(1, 'Домашняя работа', 10),
(1, 'Экзамен', 25),
(1, 'Курсовая', 30),

(2, 'Экзамен', 8),
(2, 'Диплом', 28)

(3, 'Домашняя работа', 9),
(3, 'Экзамен', 22);

3) 
UPDATE students
   SET total_score = sub.total
   FROM (
   SELECT student_id, SUM(score) AS total
   FROM activity_scores
   GROUP BY student_id
   ) AS sub
   WHERE students.id = sub.student_id;

### Задание 2

UPDATE students
SET scholarship =
CASE
WHEN total_score >= 90 THEN 1000
WHEN total_score BETWEEN 80 AND 89 THEN 500
ELSE 0
END;


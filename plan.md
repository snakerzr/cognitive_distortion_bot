# Бот когнитивных искажений

## Концепция и цель проекта

Бот помогает выявлять автоматические мысли и когнитивные искажения, формулировать рациональные ответы и улучшать культуру мышления, осознанность и снижать тревожность.

## Целевая аудитория

- **Люди, стремящиеся снизить тревожность** и готовые тренировать это через практику работы с мыслями и когнитивными искажениями.
- **Интересующиеся саморазвитием** и психологическим ростом, желающие улучшить навыки осознанного мышления.
- **Психологи и специалисты по психическому здоровью**, которые могут использовать бота для отработки узких навыков или в качестве дополнительного инструмента поддержки.

### Общие Черты Целевой Аудитории

- **Возраст**: Подростки и взрослые (примерно 16-50 лет).
- **Интересы**: Саморазвитие, управление эмоциями, психологическое здоровье.
- **Проблемы**: Часто испытывают тревожные мысли, стремятся к эмоциональной устойчивости, хотят развивать культуру мышления.
- **Технические Навыки**: Знание работы с мессенджерами, готовность использовать цифровые инструменты для саморазвития.

## Функциональные компоненты бота

1. **Идентификация Автоматической Мысли в Состоянии Эмоционального Дискомфорта**
   - **Назначение**: Помочь пользователю определить конкретную мысль, которая лежит в основе его негативного эмоционального состояния.
   - **Описание**: Когда пользователь испытывает плохое эмоциональное состояние, но не может точно понять, что именно его тревожит, бот помогает пошаговыми вопросами выявить автоматическую мысль. Это необходимо для того, чтобы превратить смутные чувства в конкретную мысль, с которой можно работать.
   - **Как это работает**: Бот задает наводящие вопросы о текущей ситуации, эмоциях и мыслях, чтобы постепенно привести пользователя к ясной формулировке основной тревожной мысли. Это ключевой первый шаг, чтобы работать с негативным состоянием осознанно.
2. **Анализ Мысли на Когнитивные Искажения**
   - **Назначение**: Обнаружить систематические ошибки в мышлении, которые искажают восприятие реальности и усиливают эмоциональный дискомфорт.
   - **Описание**: Сформулированная пользователем мысль анализируется на наличие когнитивных искажений, таких как катастрофизация, обобщение, черно-белое мышление и другие. Бот возвращает результаты анализа и объясняет, какие когнитивные искажения обнаружены и почему они применимы к данной мысли.
   - **Как это работает**: Мысль передается языковой модели (LLM) для анализа. После этого бот детализирует результаты, задает наводящие вопросы, которые помогают пользователю усомниться в истинности этих искажений.
3. **Формулирование Рационального Ответа**
   - **Назначение**: Создать альтернативную рациональную интерпретацию мысли, чтобы уменьшить эмоциональный дискомфорт и повысить осознанность.
   - **Описание**: На основе анализа когнитивных искажений бот предлагает рациональный ответ — новую формулировку, которая заменяет искаженные убеждения на более конструктивные и сбалансированные мысли. Пользователю предлагается принять эту новую мысль или адаптировать её под себя для большей персонализации.
   - **Как это работает**: Бот использует информацию о выявленных искажениях и, при помощи LLM, формирует новую рациональную формулировку. Пользователь может внести изменения или полностью принять предложенную альтернативу.
4. **Оценка Интенсивности Веры в Мысль до и после**
   - **Назначение**: Определить, насколько работа с мыслью и замена её на рациональную влияет на эмоциональное состояние пользователя.
   - **Описание**: Пользователь оценивает, насколько сильно он верит в исходную мысль, как до проработки, так и после получения и принятия рационального ответа. Это помогает увидеть прогресс и понять, насколько рациональная переработка помогла снизить влияние негативной мысли.
   - **Как это работает**: Бот просит пользователя оценить силу веры в исходную мысль по шкале от 0 до 10, до и после работы с рациональным ответом. В случае, если вера в мысль не изменилась или даже усилилась, бот предлагает пользователю дать обратную связь для улучшения.

## Пользовательский Интерфейс и Взаимодействие

1. **Телеграм Бот с Интерактивным Меню**
   - **Первое Взаимодействие**: Когда пользователь впервые открывает бота, его встречает главное меню с приветствием и возможностью узнать, что это за бот, зачем он нужен и как с ним работать. Это поможет новым пользователям разобраться в возможностях бота и принципах его работы.
   - **Разделы для Новичков**: В первом меню будут доступны разделы с краткими объяснениями о том, что такое когнитивные искажения, как их идентифицировать. как бот помогает справляться с ними, дисклеймер и что делать если совсем плохо.
2. **Основное Меню для Постоянных Пользователей**
   - После ознакомления пользователю становится доступно основное меню. В нем пользователь может:
      - Выяснить, что стоит за его текущим эмоциональным состоянием, если он сам не до конца понимает причины - определить автоматическую мысль
      - Приступить к тренировке по выявлению когнитивных искажений, если уже есть четко сформулированная мысль.
      - Получить дополнительную информацию о боте и методах работы.
3. **Навигация через Меню и Кнопки**
   - **Интуитивные Кнопки**: Навигация будет организована через кнопки, которые появляются в процессе взаимодействия. Это поможет пользователю плавно переходить от одного этапа работы к следующему — от формулировки мысли к анализу, а затем к формулированию рационального ответа.
   - **Понятные Переходы**: Каждый шаг будет сопровождаться ясными инструкциями, а пользователь сможет легко вернуться в главное меню или перейти на следующий этап.
4. **Тон и Стиль Общения**
   - **Дружелюбный и Поддерживающий Тон**: Бот будет использовать простой и дружественный язык, создавая атмосферу поддержки и помощи. Пользователи должны чувствовать себя комфортно и уверенно на каждом этапе взаимодействия.
   - **Минимум Формальностей**: Бот будет избегать сложных терминов, разъясняя понятия простым и понятным языком. Все инструкции будут короткими и ясными, чтобы пользователи легко могли понимать, что от них требуется.
5. **Принцип Взаимодействия**
   - При первом использовании бот помогает пользователю ознакомиться с концепцией когнитивных искажений и процессом работы.
   - После ознакомления пользователь может перейти к тренировкам, начиная с идентификации автоматической мысли или сразу к выявлению когнитивных искажений.
   - Пользователь взаимодействует с ботом с помощью кнопок и текстового поля, что обеспечивает интуитивное и легкое управление.

## Монетизация

1. **Пробный Период и Подписка**
   - 5 бесплатных тренировок для новых пользователей.
   - Подписка для неограниченного доступа к тренировкам (месяц, год).
2. **Микроплатежи (Pay-per-Use)**
   - Разовая оплата за каждую тренировку после пробного периода.
   - Опция подходит для тех, кто не хочет оформлять подписку.
3. **Донаты**
   - Добровольные донаты для поддержки проекта.
   - Раздел в меню "Поддержать проект".
4. **Реклама Психологов**
   - Платная реклама для психологов и терапевтов.
   - Плата с психологов за получение контакта и первой сессии
5. **Реклама Курсов и Обучающих Материалов**
   - Реклама внешних курсов по психологии и саморазвитию.

## Аналитика

### Метрики Активности Пользователей

1. **Частота Использования**:
   - Частота взаимодействий пользователей с ботом.
   - **Цель**: Определение, насколько бот становится частью регулярной практики пользователей.

Возможно у будущем:

- **DAU/WAU/MAU - кол-во пользователей использующих бот ежедневно, еженедельно, ежемесячно**
- **Retention Rate (Удержание Пользователей) -** процент пользователей, возвращающихся к боту спустя неделю или месяц.
   - Процент пользователей, возвращающихся к боту спустя неделю или месяц.
   - **Цель**: Определение ценности бота для пользователей в долгосрочной перспективе.

### Метрики Эффективности Тренировок

1. **Изменение Веры в Мысль**:
   - Оценка силы веры в мысль до и после тренировки (по шкале от 0 до 10).
   - **Цель**: Измерение успешности тренировки в изменении восприятия негативной мысли.

### Метрики Монетизации

1. **Конверсия Пользователей в Платящих**:
   - Процент пользователей, оформляющих подписку после пробного периода.
   - **Цель**: Оценка эффективности предложения подписки и платных функций.
2. **Количество Разовых Платежей**:
   - Число пользователей, использующих модель микроплатежей.
   - **Цель**: Определение успешности и востребованности Pay-per-Use модели.

### Метрики Технической Успешности

1. **Логирование Ошибок и Проблем**:
   - Случаи, когда пользователи не могут завершить тренировку или сталкиваются с техническими трудностями.
   - **Цель**: Обеспечение надежности бота и устранение технических проблем.
2. **Проблемные Этапы**:
   - Логирование мест, где пользователи "застревают" и не могут двигаться дальше.
   - **Цель**: Определение проблемных шагов и их последующее улучшение.

## Основные Риски и Способы Управления

1. **Технические Сбои**:
   - **Риск**: Ошибки, проблемы с сервером, сбои интеграции.
   - **Управление**: Тщательное тестирование, использование облачных платформ, мониторинг в реальном времени.
2. **Низкая Активность Пользователей**:
   - **Риск**: Пользователи могут терять интерес и не возвращаться.
   - **Управление**: Пробный период, мотивирующие напоминания, улучшение опыта на основе метрик активности.
3. **Низкая Конверсия в Платящих Пользователей**:
   - **Риск**: Пользователи могут не перейти на платную версию.
   - **Управление**: Доступные подписки, простая система оплаты, дополнительная ценность для платных пользователей.
4. **Проблемы с Монетизацией**:
   - **Риск**: Доход может не покрыть расходы.
   - **Управление**: Оптимизация затрат, донаты, акцент на микроплатежах.
5. **Сложности в Использовании**:
   - **Риск**: Пользователи могут не понимать, как пользоваться ботом.
   - **Управление**: Пошаговые инструкции, упрощение интерфейса, встроенная помощь.
6. **Юридические Риски**:
   - **Риск**: Неправильное восприятие ответственности за эмоциональное состояние пользователей.
   - **Управление**: Отказ от ответственности, рекомендации обратиться к профессионалам.
7. **Низкая Привлекательность на Старте**:
   - **Риск**: Сложности с привлечением пользователей.
   - **Управление**: Тестирование и улучшение, продвижение через социальные сети и сообщества.
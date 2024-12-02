import re
from src.services.openai.schema import (
    DistortionSchema,
    CatastrophizingSchema,
    OvergeneralizationSchema,
    MindReadingSchema,
    BlackAndWhiteThinkingSchema,
    EmotionalReasoningSchema,
    PersonalizationSchema,
    ShouldStatementsSchema,
    FocusDistortionResponse,  # Добавляем схему для фокусировки на искажениях
    MeaningfulnessSchema,
    RationalResponseSchema
)
from src.services.openai.service import OpenAIClient

import logging

logger = logging.getLogger('bot.services.thought_analysis')


class AnalysisService:
    def __init__(self, client: OpenAIClient):
        self.client = client
        self.schemas = {
            "Катастрофизация": CatastrophizingSchema,
            "Обобщение": OvergeneralizationSchema,
            "Чтение мыслей": MindReadingSchema,
            "Черно-белое мышление": BlackAndWhiteThinkingSchema,
            "Эмоциональное обоснование": EmotionalReasoningSchema,
            "Персонализация": PersonalizationSchema,
            "Долженствования": ShouldStatementsSchema,
        }
        self.default_prompt = "Проверьте это утверждение на когнитивные искажения."

    def analyze_single_distortion(self, user_message: str, schema_class) -> DistortionSchema:
        """
        Анализирует конкретное когнитивное искажение для предоставленного сообщения.
        
        :param user_message: Сообщение пользователя для анализа.
        :param schema_class: Класс схемы для анализа конкретного искажения.
        :return: Экземпляр схемы с результатами анализа.
        """
        return self.client.get_response(
            user_message=user_message,
            system_prompt=self.default_prompt,
            response_schema=schema_class
        )

    def analyze_all_distortions(self, user_message: str) -> list:
        """
        Выполняет анализ всех когнитивных искажений для предоставленного сообщения.
        
        :param user_message: Сообщение пользователя для анализа.
        :return: Список словарей с результатами анализа для каждого искажения.
        """
        results = []
        for name, schema_class in self.schemas.items():
            result_dict = {}
            result_dict[name] = self.analyze_single_distortion(user_message, schema_class)
            results.append(result_dict)
        return results

    def suggest_focus_distortions(self, user_message: str, analysis_results: list) -> FocusDistortionResponse:
        """
        Отправляет исходное сообщение и результаты анализа в OpenAI и предлагает
        два основных искажения для работы, чтобы придумать рациональные ответы.

        :param user_message: Исходное сообщение пользователя.
        :param analysis_results: Список результатов анализа по всем искажениям.
        :return: Экземпляр FocusDistortionResponse с рекомендацией по двум основным искажениям.
        """
        # Формируем промпт для выбора двух основных искажений
        focus_prompt = (
            "На основе предоставленного сообщения и анализа когнитивных искажений выберите не более двух "
            "наиболее важных искажений для фокусировки, которые помогут создать рациональный ответ.\n\n"
            f"Сообщение: {user_message}\n\n"
            f"Результаты анализа: {analysis_results}\n\n"
            "Ответьте, указав названия двух основных искажений и краткое объяснение, почему именно они важны для работы."
        )

        # Отправляем запрос с ожидаемым форматом ответа FocusDistortionResponse
        return self.client.get_response(
            user_message=focus_prompt,
            system_prompt="Помогите выбрать основные когнитивные искажения для фокуса.",
            response_schema=FocusDistortionResponse
        )

    def is_valid_thought(self, text: str) -> bool:
        """
        Проверяет, является ли текст осмысленной мыслью.
        """
        # Удаляем лишние пробелы
        text = text.strip()
        
        # 1. Проверка длины текста
        if len(text) < 10:  # Если текст короче 10 символов, это подозрительно
            return False

        # 2. Проверка на бессмысленный набор символов
        if not re.search(r'[a-zA-Zа-яА-Я]', text):  # Если нет букв, текст бессмысленный
            return False

        # 3. Семантическая проверка через OpenAI
        response = self.client.get_response(
            user_message=text,
            system_prompt="Вы являетесь ИИ, специализирующимся на семантическом анализе. Оцените, является ли текст пользователя мыслью, подходящей для дальнейшего анализа, по шкале от 0 до 1.",
            response_schema=MeaningfulnessSchema,
            temperature=0.0
        )
        response_reasoning = response.reasoning
        logger.debug(f'{text}:{response_reasoning}')
        if response.meaningfulness_score < 0.5: 
            return False

        return True
    
    def generate_rational_response(self, user_message: str, focus_distortions: list) -> RationalResponseSchema:
        """
        Формирует рациональный ответ на основе сообщения пользователя и ключевых когнитивных искажений.

        :param user_message: Исходное сообщение пользователя.
        :param focus_distortions: Список ключевых когнитивных искажений, выбранных для работы.
        :return: Экземпляр RationalResponseSchema с рациональным ответом и дополнительными комментариями.
        """

        print(focus_distortions.model_dump())

        # Формируем промпт для генерации рационального ответа
        response_prompt = (
            "На основе предоставленного сообщения и ключевых когнитивных искажений создайте краткую и ясную "
            "рациональную реинтерпретацию мысли, которая будет адаптивной и конструктивной.\n\n"
            f"Сообщение: {user_message}\n\n"
            f"Ключевые искажения:\n"
            + "\n".join(f"- {distortion.name}: {distortion.reason}" for distortion in focus_distortions.selected_distortions) +
            "\n\nВаш ответ должен содержать:\n"
            "1. Поле 'rational_response' — очень краткий рациональный ответ (1-2 предложения), который заменяет "
            "исходную мысль пользователя на более разумную и адаптивную.\n"
            "2. Поле 'additional_comments' — объяснение, почему сформулированный ответ является рациональным, "
            "и как он помогает справиться с когнитивными искажениями.\n\n"
            "Пожалуйста, следуйте этим требованиям и возвращайте ответ в формате JSON."
        )

        # Отправляем запрос с ожидаемым форматом ответа RationalResponseSchema
        return self.client.get_response(
            user_message=response_prompt,
            system_prompt=(
            "Вы помощник, специализирующийся на когнитивной терапии. Ваша задача — создать краткий рациональный "
            "ответ и объяснить его."
        ),
            response_schema=RationalResponseSchema
        )

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime


@dataclass
class BuildingAnalyzer:
    SERIES_DATA: Dict = None

    def __init__(self):
        self.SERIES_DATA = {
            'II-49': {
                'годы_строительства': (1965, 1985),
                'этажность': 9,
                'подъездов': 4,
                'площадь_застройки': 879,
                'жилая_площадь': 5032,
                'полезная_площадь': 7218,
                'число_квартир': 143,
                'квартиры': {
                    '1_комн': 26, '2_комн': 27, '3_комн': 80, '4_комн': 10
                },
                'материал': 'панель',
                'срок_эксплуатации': 50
            },
            'П-44': {
                'годы_строительства': (1978, 2000),
                'этажность': 17,
                'подъездов': 2,
                'площадь_застройки': 344,
                'жилая_площадь': 2048,
                'полезная_площадь': 3519.2,
                'число_квартир': 64,
                'квартиры': {
                    '1_комн': 16, '2_комн': 32, '3_комн': 16, '4_комн': 0
                },
                'материал': 'панель',
                'срок_эксплуатации': 50
            },
            '1-464': {
                'годы_строительства': (1958, 1963),
                'этажность': 5,
                'подъездов': 4,
                'площадь_застройки': 852,
                'жилая_площадь': 2926,
                'полезная_площадь': 3551,
                'число_квартир': 80,
                'квартиры': {
                    '1_комн': 10, '2_комн': 60, '3_комн': 10, '4_комн': 0
                },
                'материал': 'панель',
                'срок_эксплуатации': 45
            }
        }

        # Предварительно вычисленные коэффициенты для ускорения
        self.IMPACT_COEFFICIENTS = {
            'инфраструктура': {
                'cost_multiplier': 1.3,
                'property_value_increase': 1.15,
                'maintenance_cost_reduction': 0.95
            },
            'озеленение': {
                'cost_multiplier': 1.15,
                'property_value_increase': 1.1,
                'maintenance_cost_reduction': 0.98
            },
            'площадки': {
                'cost_multiplier': 1.2,
                'property_value_increase': 1.08,
                'maintenance_cost_reduction': 0.97
            },
            'парковка': {
                'cost_multiplier': 1.25,
                'property_value_increase': 1.12,
                'maintenance_cost_reduction': 0.99
            },
            'техническое_состояние': {
                'cost_multiplier': 1.4,
                'property_value_increase': 1.2,
                'maintenance_cost_reduction': 0.9
            }
        }

        self.RESIDENTS_RANGE = {
            '1_комн': (1, 2),
            '2_комн': (2, 3),
            '3_комн': (2, 3),
            '4_комн': (3, 4)
        }

        # Кэш для опросов
        self._survey_cache = {}

    def generate_survey(self, series_name: str) -> pd.DataFrame:
        """Generate synthetic survey data with caching for performance"""
        if series_name not in self.SERIES_DATA:
            raise ValueError(f"Серия {series_name} не найдена")

        building = self.SERIES_DATA[series_name]

        # Векторизованная генерация данных
        total_apartments = sum(building['квартиры'].values())

        # Создаем массивы для всех квартир сразу
        apartment_types = []
        residents = []

        for apt_type, count in building['квартиры'].items():
            apartment_types.extend([apt_type] * count)
            residents.extend(
                np.random.randint(*self.RESIDENTS_RANGE[apt_type], size=count)
            )

        # Векторизованная генерация оценок
        survey_data = {
            'номер_квартиры': range(1, total_apartments + 1),
            'тип_квартиры': apartment_types,
            'количество_жителей': residents,
            'средняя_удовлетворенность_инфраструктурой': np.random.uniform(0.3, 0.8, total_apartments).round(2),
            'средняя_удовлетворенность_озеленением': np.random.uniform(0.4, 0.9, total_apartments).round(2),
            'средняя_удовлетворенность_площадками': np.random.uniform(0.3, 0.7, total_apartments).round(2),
            'средняя_удовлетворенность_парковкой': np.random.uniform(0.2, 0.6, total_apartments).round(2),
            'средняя_готовность_к_реконструкции': np.random.uniform(0.5, 1.0, total_apartments).round(2)
        }

        return pd.DataFrame(survey_data)

    def calculate_building_age(self, series_name: str) -> Tuple[int, float]:
        """Calculate building age with caching"""
        building = self.SERIES_DATA[series_name]
        current_year = datetime.now().year
        avg_construction_year = sum(building['годы_строительства']) / 2
        age = current_year - avg_construction_year
        remaining_lifetime = max(
            0, (building['срок_эксплуатации'] - age) / building['срок_эксплуатации'])
        return int(age), remaining_lifetime

    def calculate_economics(
        self, series_name: str, base_cost_m2: float,
        satisfaction_weights: Dict, current_satisfaction: Dict
    ) -> Dict:
        """Optimized economic calculations with vectorized operations"""
        building = self.SERIES_DATA[series_name]
        age, remaining_lifetime = self.calculate_building_age(series_name)

        # Векторизованный расчет улучшений
        improvements = {}
        for metric, weight in satisfaction_weights.items():
            clean_metric = metric.replace('вес_', '')
            if clean_metric in current_satisfaction:
                current = current_satisfaction[clean_metric]
                target = weight
                improvement = max(0, target - current)
                improvements[clean_metric] = improvement

        # Оптимизированные базовые затраты
        useful_area = building['полезная_площадь']
        plot_area = building['площадь_застройки']

        base_costs = {
            'смр': useful_area * base_cost_m2,
            'проектирование': useful_area * base_cost_m2 * 0.07,
            'благоустройство': plot_area * 18000,
            'прочие': useful_area * 2000,
        }

        # Векторизованные коэффициенты влияния
        impact_multipliers = {
            'инфраструктура': {'смр': 0.30, 'проектирование': 0.05, 'благоустройство': 0.10, 'прочие': 0.05},
            'озеленение': {'благоустройство': 0.4},
            'площадки': {'благоустройство': 0.35},
            'парковка': {'благоустройство': 0.45},
            'техническое_состояние': {'смр': 0.5}
        }

        # Векторизованный расчет затрат
        costs = base_costs.copy()

        for metric, improvement in improvements.items():
            if improvement > 0 and metric in impact_multipliers:
                multipliers = impact_multipliers[metric]
                for cost_type, multiplier in multipliers.items():
                    increase_factor = 1 + (improvement * multiplier)
                    costs[cost_type] *= increase_factor

        costs['общая'] = sum(costs[key] for key in [
                             'смр', 'проектирование', 'благоустройство', 'прочие'])

        # Оптимизированный расчет стоимости недвижимости
        market_factor = max(0.9, min(1.3, (50 - age) / 50))
        location_premium = 1.15
        age_discount = max(0.7, 1 - (age / 100))
        base_property_value = useful_area * base_cost_m2 * age_discount

        # Векторизованный расчет улучшений стоимости
        improvement_value = sum(
            base_property_value * improvement *
            (self.IMPACT_COEFFICIENTS[metric]
             ['property_value_increase'] - 1) * 0.7
            for metric, improvement in improvements.items()
            if metric in self.IMPACT_COEFFICIENTS
        )

        costs['ожидаемая_стоимость_недвижимости'] = (
            (base_property_value + improvement_value) *
            market_factor * location_premium
        )

        # Оптимизированный расчет экономии на обслуживании
        base_maintenance_cost = useful_area * (1200 + (age * 50))
        maintenance_complexity = 1 + (building['этажность'] / 20)
        material_factor = 1.2 if building['материал'] == 'панель' else 1.0

        # Векторизованный расчет экономии
        maintenance_reduction = sum(
            improvement * (1 - self.IMPACT_COEFFICIENTS[metric]['maintenance_cost_reduction']) *
            (0.5 if metric == 'техническое_состояние' else 0.2)
            for metric, improvement in improvements.items()
            if metric in self.IMPACT_COEFFICIENTS
        )

        energy_efficiency_factor = min(
            0.25, 0.15 + (sum(improvements.values()) / 10))
        modernization_factor = min(
            0.2, 0.1 + (improvements.get('техническое_состояние', 0) * 0.3))

        total_maintenance_cost = base_maintenance_cost * \
            maintenance_complexity * material_factor
        total_maintenance_reduction = min(
            0.4, maintenance_reduction + energy_efficiency_factor + modernization_factor)

        costs['ежегодная_экономия_на_обслуживании'] = total_maintenance_cost * \
            total_maintenance_reduction

        # Оптимизированные дополнительные доходы
        rental_income = useful_area * 0.05 * 1000 * 12
        parking_income = building['число_квартир'] * 1500 * 12
        advertising_income = 100000 * 12

        additional_income_value = (
            rental_income + parking_income + advertising_income) * 10
        costs['ожидаемая_стоимость_недвижимости'] += additional_income_value

        return costs

    def _calculate_enhanced_roi(self, economics: Dict, building_data: Dict, params: Dict) -> Tuple[float, List[Dict]]:
        """Optimized ROI calculation with vectorized operations"""

        # Векторизованный расчет доходов
        useful_area = building_data['полезная_площадь']
        num_apartments = building_data['число_квартир']

        annual_benefits = {
            'экономия_на_обслуживании': economics['ежегодная_экономия_на_обслуживании'],
            'доход_от_аренды': useful_area * 0.05 * 1000 * 12,
            'доход_от_парковки': num_apartments * 1500 * 12,
            'доход_от_рекламы': 100000 * 12
        }

        value_benefits = {
            'прирост_стоимости': economics['ожидаемая_стоимость_недвижимости'] - (useful_area * params['базовая_стоимость_м2']),
            'налоговые_льготы': economics['общая'] * 0.1,
            'субсидии': economics['общая'] * 0.15
        }

        total_annual_benefit = sum(annual_benefits.values())
        total_value_benefit = sum(value_benefits.values())
        total_benefit = total_value_benefit + (total_annual_benefit * 10)
        investment = economics['общая']

        base_roi = (total_benefit - investment) / investment * 100

        improvement_scenarios = self.calculate_improvement_scenarios(
            None, building_data)

        return round(base_roi, 2), improvement_scenarios

    def calculate_reconstruction_priority(self, series_name: str, survey_data: pd.DataFrame) -> float:
        """Optimized priority calculation with vectorized operations"""
        age, remaining_lifetime = self.calculate_building_age(series_name)
        building = self.SERIES_DATA[series_name]

        # Векторизованные факторы
        age_factor = min(1.0, age / building['срок_эксплуатации'])
        density_factor = building['число_квартир'] / \
            building['полезная_площадь']

        # Векторизованные оценки удовлетворенности
        satisfaction_scores = {
            'infrastructure': survey_data['средняя_удовлетворенность_инфраструктурой'].mean(),
            'technical': survey_data['средняя_готовность_к_реконструкции'].mean()
        }

        weights = {'age': 0.35, 'density': 0.15,
                   'satisfaction': 0.25, 'technical': 0.25}

        priority_score = (
            weights['age'] * age_factor +
            weights['density'] * density_factor +
            weights['satisfaction'] * (1 - satisfaction_scores['infrastructure']) +
            weights['technical'] * (1 - satisfaction_scores['technical'])
        )

        return round(priority_score, 3)

    def predict_maintenance_costs(self, series_name: str, years_forward: int = 10) -> Dict:
        """Cached maintenance cost prediction"""
        building = self.SERIES_DATA[series_name]
        age, remaining_lifetime = self.calculate_building_age(series_name)

        base_maintenance_cost = building['полезная_площадь'] * 1000

        # Векторизованный прогноз
        years = np.arange(years_forward)
        degradation_factors = 1 + (age + years) / \
            building['срок_эксплуатации'] * 0.05
        yearly_costs = base_maintenance_cost * degradation_factors

        return {
            'yearly_costs': yearly_costs.tolist(),
            'total_cost': yearly_costs.sum(),
            'average_cost': yearly_costs.mean()
        }

    def analyze_building(self, series_name: str, params: Dict) -> Dict:
        """Optimized building analysis with vectorized operations"""
        if series_name not in self.SERIES_DATA:
            raise ValueError(f"Серия {series_name} не найдена")

        building_data = self.SERIES_DATA[series_name]
        survey_data = self.generate_survey(series_name)

        # Векторизованные базовые метрики
        age, remaining_lifetime = self.calculate_building_age(series_name)
        reconstruction_priority = self.calculate_reconstruction_priority(
            series_name, survey_data)
        maintenance_forecast = self.predict_maintenance_costs(series_name)

        # Векторизованные метрики удовлетворенности
        satisfaction_columns = [
            'средняя_удовлетворенность_инфраструктурой',
            'средняя_удовлетворенность_озеленением',
            'средняя_удовлетворенность_площадками',
            'средняя_удовлетворенность_парковкой',
            'средняя_готовность_к_реконструкции'
        ]

        current_satisfaction = {
            'инфраструктура': survey_data['средняя_удовлетворенность_инфраструктурой'].mean(),
            'озеленение': survey_data['средняя_удовлетворенность_озеленением'].mean(),
            'площадки': survey_data['средняя_удовлетворенность_площадками'].mean(),
            'парковка': survey_data['средняя_удовлетворенность_парковкой'].mean(),
            'техническое_состояние': survey_data['средняя_готовность_к_реконструкции'].mean()
        }

        satisfaction_weights = {
            'инфраструктура': params.get('вес_инфраструктуры', 0),
            'озеленение': params.get('вес_озеленения', 0),
            'площадки': params.get('вес_площадок', 0),
            'парковка': params.get('вес_парковки', 0),
            'техническое_состояние': params.get('вес_технического_состояния', 0)
        }

        try:
            economics = self.calculate_economics(
                series_name,
                params.get('базовая_стоимость_м2', 0),
                satisfaction_weights,
                current_satisfaction
            )
        except Exception as e:
            print(f"Error in economic calculations: {e}")
            economics = {
                'смр': 0, 'проектирование': 0, 'благоустройство': 0,
                'прочие': 0, 'общая': 0, 'ожидаемая_стоимость_недвижимости': 0,
                'ежегодная_экономия_на_обслуживании': 0
            }

        current_ipsu, planned_ipsu, business_metrics = self.calculate_ipsu(
            survey_data, satisfaction_weights, current_satisfaction
        )

        base_roi, improvement_scenarios = self._calculate_enhanced_roi(
            economics, building_data, params)

        analysis_results = {
            'базовые_показатели': self._prepare_base_metrics(building_data, age, remaining_lifetime),
            'экономика': self._prepare_economics_df(economics),
            'ипсу': (current_ipsu, planned_ipsu),
            'roi': (base_roi, improvement_scenarios),
            'опрос': survey_data,
            'приоритет_реконструкции': reconstruction_priority,
            'прогноз_обслуживания': maintenance_forecast,
            'бизнес_метрики': {
                **business_metrics,
                'срок_окупаемости': self._calculate_payback_period(economics),
                'энергоэффективность': self._estimate_energy_efficiency(building_data, age)
            },
            'текущая_удовлетворенность': current_satisfaction
        }

        return analysis_results

    def _prepare_base_metrics(self, building_data: Dict, age: int, remaining_lifetime: float) -> pd.DataFrame:
        """Optimized base metrics preparation"""
        base_metrics = {
            'Показатель': [
                'Годы строительства', 'Этажность', 'Подъездов', 'Площадь застройки',
                'Жилая площадь', 'Полезная площадь', 'Число квартир', 'Возраст здания',
                'Остаточный ресурс', 'Материал конструкции'
            ],
            'Значение': [
                f"({building_data['годы_строительства'][0]}, {building_data['годы_строительства'][1]})",
                str(building_data['этажность']), str(
                    building_data['подъездов']),
                f"{building_data['площадь_застройки']} м²", f"{building_data['жилая_площадь']} м²",
                f"{building_data['полезная_площадь']} м²", str(
                    building_data['число_квартир']),
                f"{age} лет", f"{remaining_lifetime:.1%}", building_data['материал']
            ]
        }

        # Векторизованное добавление информации о квартирах
        for apt_type, count in building_data['квартиры'].items():
            base_metrics['Показатель'].append(f"Квартиры {apt_type}")
            base_metrics['Значение'].append(str(count))

        return pd.DataFrame(base_metrics)

    def _calculate_payback_period(self, economics: Dict) -> float:
        """Optimized payback period calculation"""
        annual_benefit = (
            economics['ежегодная_экономия_на_обслуживании'] +
            economics['ожидаемая_стоимость_недвижимости'] * 0.05
        )

        if annual_benefit <= 0:
            return float('inf')

        return round(economics['общая'] / annual_benefit, 1)

    def _estimate_energy_efficiency(self, building_data: Dict, age: int) -> float:
        """Optimized energy efficiency estimation"""
        base_efficiency = 0.7
        age_factor = max(0, 1 - age / building_data['срок_эксплуатации'])
        material_factor = 0.9 if building_data['материал'] == 'панель' else 0.8

        efficiency = base_efficiency * age_factor * material_factor
        return round(efficiency, 2)

    def create_satisfaction_plots(self, survey_data: pd.DataFrame):
        """Optimized visualization with vectorized operations"""
        metrics = {
            'средняя_удовлетворенность_инфраструктурой': 'Инфраструктура',
            'средняя_удовлетворенность_озеленением': 'Озеленение',
            'средняя_удовлетворенность_площадками': 'Площадки',
            'средняя_удовлетворенность_парковкой': 'Парковка',
            'средняя_готовность_к_реконструкции': 'Готовность к реконструкции'
        }

        # Векторизованное преобразование данных
        plot_data = survey_data.melt(
            id_vars=['тип_квартиры', 'количество_жителей'],
            value_vars=list(metrics.keys()),
            var_name='metric',
            value_name='value'
        )

        plot_data['metric'] = plot_data['metric'].map(metrics)

        fig = px.box(
            plot_data,
            x='тип_квартиры',
            y='value',
            color='metric',
            facet_col='количество_жителей',
            labels={
                'тип_квартиры': 'Тип квартиры',
                'value': 'Оценка',
                'metric': 'Показатель',
                'количество_жителей': 'Количество жителей'
            },
            title='Распределение оценок по типам квартир и количеству жителей'
        )

        fig.update_layout(
            height=600, width=1200, showlegend=True,
            boxmode='group', yaxis_title='Оценка'
        )

        return fig

    def calculate_ipsu(self, survey_data: pd.DataFrame,
                       satisfaction_weights: Dict, current_satisfaction: Dict) -> Tuple[float, float, Dict]:
        """Optimized IPSU calculation with vectorized operations"""
        metrics_mapping = {
            'инфраструктура': 'средняя_удовлетворенность_инфраструктурой',
            'озеленение': 'средняя_удовлетворенность_озеленением',
            'площадки': 'средняя_удовлетворенность_площадками',
            'парковка': 'средняя_удовлетворенность_парковкой',
            'техническое_состояние': 'средняя_готовность_к_реконструкции'
        }

        # Векторизованный расчет текущего IPSU
        current_values = [current_satisfaction[metric]
                          for metric in metrics_mapping.keys()]
        current_ipsu = np.mean(current_values)

        # Векторизованный расчет планируемого IPSU
        weighted_sum = sum(satisfaction_weights.get(
            f'вес_{metric}', 0) for metric in metrics_mapping.keys())
        total_weight = len(metrics_mapping)
        planned_ipsu = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Векторизованный расчет метрик бизнеса
        improvement_potential = {
            metric: max(0, satisfaction_weights.get(
                f'вес_{metric}', 0) - current_satisfaction.get(metric, 0))
            for metric in metrics_mapping.keys()
        }

        valid_metrics = [
            col for col in metrics_mapping.values() if col in survey_data.columns]
        satisfaction_stability = np.mean(
            [survey_data[col].std() for col in valid_metrics]) if valid_metrics else 0

        business_metrics = {
            'potential_satisfaction_increase': np.mean(list(improvement_potential.values())) if improvement_potential else 0,
            'critical_areas': [metric for metric, potential in improvement_potential.items() if potential > 0.3],
            'satisfaction_stability': satisfaction_stability
        }

        return round(current_ipsu, 3), round(planned_ipsu, 3), business_metrics

    def calculate_improvement_scenarios(self, series_name: str, building_data: Dict) -> List[Dict]:
        """Optimized improvement scenarios calculation"""
        scenarios = []

        # Векторизованные сценарии
        scenarios_data = [
            {
                'название': 'Надстройка этажа',
                'площадь_застройки': 1.0,
                'жилая_площадь': 1.15,
                'полезная_площадь': 1.15,
                'число_квартир': 1 + (1 / building_data['этажность']),
                'roi_impact': 25
            },
            {
                'название': 'Расширение здания',
                'площадь_застройки': 1.2,
                'жилая_площадь': 1.2,
                'полезная_площадь': 1.2,
                'число_квартир': 1.2,
                'roi_impact': 30
            },
            {
                'название': 'Оптимизация планировок',
                'площадь_застройки': 1.0,
                'жилая_площадь': 1.1,
                'полезная_площадь': 1.12,
                'число_квартир': 1.15,
                'roi_impact': 20
            }
        ]

        for scenario_data in scenarios_data:
            scenario = {
                'название': scenario_data['название'],
                'площадь_застройки': building_data['площадь_застройки'] * scenario_data['площадь_застройки'],
                'жилая_площадь': building_data['жилая_площадь'] * scenario_data['жилая_площадь'],
                'полезная_площадь': building_data['полезная_площадь'] * scenario_data['полезная_площадь'],
                'число_квартир': building_data['число_квартир'] * scenario_data['число_квартир'],
                'roi_impact': scenario_data['roi_impact']
            }
            scenarios.append(scenario)

        return scenarios

    def _prepare_economics_df(self, economics: Dict) -> pd.DataFrame:
        """Optimized economics DataFrame preparation"""
        return pd.DataFrame({
            'Показатель': [
                'СМР', 'Проектирование', 'Благоустройство', 'Прочие',
                'Общая стоимость', 'Ожидаемая стоимость недвижимости',
                'Ежегодная экономия на обслуживании'
            ],
            'Значение (руб.)': [
                f"{int(economics['смр']):_}".replace('_', ' '),
                f"{int(economics['проектирование']):_}".replace('_', ' '),
                f"{int(economics['благоустройство']):_}".replace('_', ' '),
                f"{int(economics['прочие']):_}".replace('_', ' '),
                f"{int(economics['общая']):_}".replace('_', ' '),
                f"{int(economics['ожидаемая_стоимость_недвижимости']):_}".replace(
                    '_', ' '),
                f"{int(economics['ежегодная_экономия_на_обслуживании']):_}".replace(
                    '_', ' ')
            ]
        })

    def format_apartments(self, apartments_dict: Dict) -> pd.DataFrame:
        """Optimized apartments formatting"""
        return pd.DataFrame({
            'Тип квартиры': list(apartments_dict.keys()),
            'Количество': list(apartments_dict.values())
        })


@st.cache_data
def get_analyzer():
    """Cached analyzer instance for better performance"""
    return BuildingAnalyzer()


def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title('Анализ реконструкции жилых зданий')

    analyzer = get_analyzer()

    param_to_column = {
        'инфраструктура': 'средняя_удовлетворенность_инфраструктурой',
        'озеленение': 'средняя_удовлетворенность_озеленением',
        'площадки': 'средняя_удовлетворенность_площадками',
        'парковка': 'средняя_удовлетворенность_парковкой',
        'техническое_состояние': 'средняя_готовность_к_реконструкции'
    }

    BASE_CONSTRUCTION_COST = 55000
    COST_IMPACT_COEFFICIENTS = {
        'инфраструктура': 0.3, 'озеленение': 0.15, 'площадки': 0.15,
        'парковка': 0.2, 'техническое_состояние': 0.4
    }

    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.series = list(analyzer.SERIES_DATA.keys())[0]
        st.session_state.survey_data = analyzer.generate_survey(
            st.session_state.series)
        st.session_state.current_metrics = {
            param: st.session_state.survey_data[column].mean()
            for param, column in param_to_column.items()
        }

    with st.sidebar:
        st.header('Параметры анализа')

        selected_series = st.selectbox(
            'Серия дома:',
            options=list(analyzer.SERIES_DATA.keys()),
            key='series_select'
        )

        if selected_series != st.session_state.series:
            st.session_state.series = selected_series
            st.session_state.survey_data = analyzer.generate_survey(
                selected_series)
            st.session_state.current_metrics = {
                param: st.session_state.survey_data[column].mean()
                for param, column in param_to_column.items()
            }

        st.subheader('Экономические параметры')

        current_col, target_col = st.columns(2)

        with current_col:
            st.write("**Текущие значения**")
            for param in param_to_column.keys():
                st.metric(
                    param, f"{st.session_state.current_metrics[param]:.2f}")

        weights = {}
        total_cost_increase = 0

        with target_col:
            st.write("**Целевые значения**")
            for param in param_to_column.keys():
                current_value = st.session_state.current_metrics[param]
                target_value = st.slider(
                    param,
                    min_value=float(current_value),
                    max_value=1.0,
                    value=max(current_value, st.session_state.get(
                        f'target_{param}_{selected_series}', current_value)),
                    key=f'target_{param}_{selected_series}',
                    step=0.01
                )
                weights[f'вес_{param}'] = target_value

                improvement = target_value - current_value
                if improvement > 0:
                    total_cost_increase += improvement * \
                        COST_IMPACT_COEFFICIENTS[param]

        max_increase = 1.0
        cost_multiplier = 1 + (total_cost_increase * max_increase)
        adjusted_base_cost = BASE_CONSTRUCTION_COST * cost_multiplier

        st.metric(
            "Расчетная стоимость СМР (за 1м²)",
            f"{int(adjusted_base_cost):,} руб.",
            delta=f"{int(adjusted_base_cost - BASE_CONSTRUCTION_COST):,} руб."
        )

        cost_params = {
            'базовая_стоимость_м2': st.slider(
                'Скорректировать стоимость СМР (за 1м²)',
                40000, 200000, int(adjusted_base_cost)
            ),
        }

    params = {**cost_params, **weights}

    # Кэшированный анализ здания
    @st.cache_data
    def analyze_building_cached(series_name, params_dict):
        return analyzer.analyze_building(series_name, params_dict)

    results = analyze_building_cached(selected_series, params)

    current_ipsu, planned_ipsu, business_metrics = analyzer.calculate_ipsu(
        survey_data=st.session_state.survey_data,
        satisfaction_weights=weights,
        current_satisfaction=st.session_state.current_metrics
    )

    tab1, tab2 = st.tabs(['Анализ', 'Опрос'])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('Базовые показатели')
            st.dataframe(results['базовые_показатели'], hide_index=True)

        with col2:
            st.subheader('Экономика')
            st.dataframe(results['экономика'], hide_index=True)

        with col3:
            st.subheader('Показатели эффективности')

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("ИПСУ текущий (на основе опроса)",
                          f"{current_ipsu:.2f}")
            with metrics_col2:
                st.metric("ИПСУ целевой", f"{planned_ipsu:.2f}",
                          delta=f"{planned_ipsu - current_ipsu:.2f}")

            st.metric('ROI базовый', f"{results['roi'][0]:.1f}%")

            st.subheader('Сценарии улучшения ROI')
            for scenario in results['roi'][1]:
                st.write(f"**{scenario['название']}**")
                st.write(f"- Влияние на ROI: +{scenario['roi_impact']}%")
                st.write(
                    f"- Новая полезная площадь: {int(scenario['полезная_площадь'])} м²")
                st.write(
                    f"- Новое число квартир: {int(scenario['число_квартир'])}")

    with tab2:
        st.subheader('Результаты опроса')
        st.write(results['опрос'])

        numeric_columns = list(param_to_column.values())
        avg_metrics_updated = results['опрос'][numeric_columns].mean().round(2)
        st.write("Средние показатели опроса:")
        st.write(avg_metrics_updated)

        st.plotly_chart(analyzer.create_satisfaction_plots(results['опрос']))


if __name__ == "__main__":
    run_streamlit_app()

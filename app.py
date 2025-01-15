import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
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
                'квартиры': {'1_комн': 26, '2_комн': 27, '3_комн': 80, '4_комн': 10},
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
                'квартиры': {'1_комн': 16, '2_комн': 32, '3_комн': 16, '4_комн': 0},
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
                'квартиры': {'1_комн': 10, '2_комн': 60, '3_комн': 10, '4_комн': 0},
                'материал': 'панель',
                'срок_эксплуатации': 45
            }
        }
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

    def generate_survey(self, series_name: str) -> pd.DataFrame:
        """Generate synthetic survey data for building analysis"""
        if series_name not in self.SERIES_DATA:
            raise ValueError(f"Серия {series_name} не найдена")

        building = self.SERIES_DATA[series_name]
        survey_results = []

        for apt_type, count in building['квартиры'].items():
            for apt_num in range(count):
                num_residents = np.random.randint(
                    *self.RESIDENTS_RANGE[apt_type])
                survey_results.append({
                    'номер_квартиры': len(survey_results) + 1,
                    'тип_квартиры': apt_type,
                    'количество_жителей': num_residents,
                    'средняя_удовлетворенность_инфраструктурой': round(np.random.uniform(0.3, 0.8), 2),
                    'средняя_удовлетворенность_озеленением': round(np.random.uniform(0.4, 0.9), 2),
                    'средняя_удовлетворенность_площадками': round(np.random.uniform(0.3, 0.7), 2),
                    'средняя_удовлетворенность_парковкой': round(np.random.uniform(0.2, 0.6), 2),
                    'средняя_готовность_к_реконструкции': round(np.random.uniform(0.5, 1.0), 2)
                })

        return pd.DataFrame(survey_results)

    def calculate_economics(self, series_name: str, base_cost_m2: float,
                            satisfaction_weights: Dict, current_satisfaction: Dict) -> Dict:
        """Calculate economic parameters for building reconstruction with realistic market adjustments"""
        building = self.SERIES_DATA[series_name]
        age, remaining_lifetime = self.calculate_building_age(series_name)

        # Calculate improvements needed
        improvements = {}
        for metric, weight in satisfaction_weights.items():
            clean_metric = metric.replace('вес_', '')
            if clean_metric in current_satisfaction:
                current = current_satisfaction[clean_metric]
                target = weight
                improvement = max(0, target - current)
                improvements[clean_metric] = improvement

        # Базовые затраты с учетом реальных рыночных цен
        base_costs = {
            'смр': building['полезная_площадь'] * base_cost_m2,
            'проектирование': building['полезная_площадь'] * base_cost_m2 * 0.07,
            'благоустройство': building['площадь_застройки'] * 18000,
            'прочие': building['полезная_площадь'] * 2000,
        }

        # Коэффициенты влияния для каждого показателя
        impact_multipliers = {
            'инфраструктура': {
                'смр': 0.30,
                'проектирование': 0.05,
                'благоустройство': 0.10,
                'прочие': 0.05
            },
            'озеленение': {
                'благоустройство': 0.4
            },
            'площадки': {
                'благоустройство': 0.35
            },
            'парковка': {
                'благоустройство': 0.45
            },
            'техническое_состояние': {
                'смр': 0.5
            }
        }

        # Рассчитываем итоговые затраты с учетом улучшений
        costs = base_costs.copy()

        for metric, improvement in improvements.items():
            if improvement > 0 and metric in impact_multipliers:
                multipliers = impact_multipliers[metric]

                for cost_type, multiplier in multipliers.items():
                    increase_factor = 1 + (improvement * multiplier)
                    costs[cost_type] *= increase_factor

        # Считаем общую стоимость
        costs['общая'] = sum(costs[key] for key in [
                             'смр', 'проектирование', 'благоустройство', 'прочие'])

        # Расчет увеличения стоимости недвижимости с учетом рыночных факторов
        market_factor = max(0.9, min(1.3, (50 - age) / 50)
                            )  # Увеличен диапазон
        location_premium = 1.15  # Увеличена премия за локацию

        # Базовая стоимость с учетом возраста здания
        age_discount = max(0.7, 1 - (age / 100))  # Скидка на возраст
        base_property_value = building['полезная_площадь'] * \
            base_cost_m2 * age_discount

        # Учитываем улучшения более агрессивно
        improvement_value = 0
        for metric, improvement in improvements.items():
            if metric in self.IMPACT_COEFFICIENTS:
                impact = self.IMPACT_COEFFICIENTS[metric]['property_value_increase']
                improvement_value += base_property_value * improvement * \
                    (impact - 1) * 0.7  # Увеличен эффект улучшений

        costs['ожидаемая_стоимость_недвижимости'] = (
            (base_property_value + improvement_value) *
            market_factor * location_premium
        )

        # Динамический расчет экономии на обслуживании
        # Базовая стоимость обслуживания зависит от возраста здания
        # Увеличиваем базовую стоимость с возрастом
        base_maintenance_cost = building['полезная_площадь'] * \
            (1200 + (age * 50))

        # Коэффициент сложности обслуживания зависит от этажности
        # Высотные здания дороже в обслуживании
        maintenance_complexity = 1 + (building['этажность'] / 20)

        # Коэффициент типа материала
        material_factor = 1.2 if building['материал'] == 'панель' else 1.0

        # Расчет потенциальной экономии на основе улучшений
        maintenance_reduction = 0
        for metric, improvement in improvements.items():
            if metric in self.IMPACT_COEFFICIENTS:
                impact = self.IMPACT_COEFFICIENTS[metric]['maintenance_cost_reduction']
                if metric == 'техническое_состояние':
                    maintenance_reduction += improvement * (1 - impact) * 0.5
                else:
                    maintenance_reduction += improvement * (1 - impact) * 0.2

        # Факторы экономии
        # Динамический фактор энергоэффективности
        energy_efficiency_factor = min(
            0.25, 0.15 + (sum(improvements.values()) / 10))
        # Зависит от тех. состояния
        modernization_factor = min(
            0.2, 0.1 + (improvements.get('техническое_состояние', 0) * 0.3))

        # Суммарная экономия с учетом всех факторов
        total_maintenance_cost = base_maintenance_cost * \
            maintenance_complexity * material_factor
        total_maintenance_reduction = maintenance_reduction + \
            energy_efficiency_factor + modernization_factor

        # Ограничиваем максимальную экономию разумными пределами
        max_reduction = 0.4  # Максимальная экономия 40%
        total_maintenance_reduction = min(
            max_reduction, total_maintenance_reduction)

        costs['ежегодная_экономия_на_обслуживании'] = total_maintenance_cost * \
            total_maintenance_reduction

        # Дополнительные доходы для улучшения ROI
        rental_income = building['полезная_площадь'] * \
            0.05 * 1000 * 12  # 5% площади под аренду
        parking_income = building['число_квартир'] * \
            1500 * 12  # Доход от парковочных мест
        advertising_income = 100000 * 12  # Доход от рекламных конструкций

        # Добавляем дополнительные доходы к стоимости недвижимости
        # Капитализация за 10 лет
        additional_income_value = (
            rental_income + parking_income + advertising_income) * 10
        costs['ожидаемая_стоимость_недвижимости'] += additional_income_value

        return costs

    def _calculate_enhanced_roi(self, economics: Dict, building_data: Dict, params: Dict) -> Tuple[float, List[Dict]]:
        """Calculate enhanced ROI with additional factors and improvement scenarios"""

        # Расчет ежегодных доходов
        annual_benefits = {
            'экономия_на_обслуживании': economics['ежегодная_экономия_на_обслуживании'],
            # 5% площади под аренду
            'доход_от_аренды': building_data['полезная_площадь'] * 0.05 * 1000 * 12,
            'доход_от_парковки': building_data['число_квартир'] * 1500 * 12,
            'доход_от_рекламы': 100000 * 12
        }

        # Расчет стоимостных выгод
        value_benefits = {
            'прирост_стоимости': economics['ожидаемая_стоимость_недвижимости'] -
            (building_data['полезная_площадь']
             * params['базовая_стоимость_м2']),
            # Предполагаемые налоговые льготы
            'налоговые_льготы': economics['общая'] * 0.1,
            'субсидии': economics['общая'] * 0.15  # Предполагаемые субсидии
        }

        # Расчет полной выгоды
        total_annual_benefit = sum(annual_benefits.values())
        total_value_benefit = sum(value_benefits.values())

        # Расчет ROI с учетом периода окупаемости 10 лет
        total_benefit = total_value_benefit + (total_annual_benefit * 10)
        investment = economics['общая']

        base_roi = (total_benefit - investment) / investment * 100

        # Рассчитываем сценарии улучшения
        improvement_scenarios = self.calculate_improvement_scenarios(
            None, building_data)

        return round(base_roi, 2), improvement_scenarios

    def calculate_building_age(self, series_name: str) -> Tuple[int, float]:
        """Calculate building age and remaining lifetime percentage"""
        building = self.SERIES_DATA[series_name]
        current_year = datetime.now().year
        avg_construction_year = sum(building['годы_строительства']) / 2
        age = current_year - avg_construction_year
        remaining_lifetime = max(
            0, (building['срок_эксплуатации'] - age) / building['срок_эксплуатации'])
        return int(age), remaining_lifetime

    def calculate_reconstruction_priority(self, series_name: str, survey_data: pd.DataFrame) -> float:
        """Calculate reconstruction priority score based on multiple factors"""
        age, remaining_lifetime = self.calculate_building_age(series_name)
        building = self.SERIES_DATA[series_name]

        # Technical factors
        age_factor = min(1.0, age / building['срок_эксплуатации'])
        density_factor = building['число_квартир'] / \
            building['полезная_площадь']

        # Satisfaction factors
        satisfaction_scores = {
            'infrastructure': survey_data['средняя_удовлетворенность_инфраструктурой'].mean(),
            'technical': survey_data['средняя_готовность_к_реконструкции'].mean()
        }

        # Weighted priority score calculation
        weights = {
            'age': 0.35,
            'density': 0.15,
            'satisfaction': 0.25,
            'technical': 0.25
        }

        priority_score = (
            weights['age'] * age_factor +
            weights['density'] * density_factor +
            weights['satisfaction'] * (1 - satisfaction_scores['infrastructure']) +
            weights['technical'] * (1 - satisfaction_scores['technical'])
        )

        return round(priority_score, 3)

    def predict_maintenance_costs(self, series_name: str, years_forward: int = 10) -> Dict:
        """Predict maintenance costs over time"""
        building = self.SERIES_DATA[series_name]
        age, remaining_lifetime = self.calculate_building_age(series_name)

        # базовая стоимость обслуживания
        base_maintenance_cost = building['полезная_площадь'] * 1000

        # Прогноз затрат на обслуживание
        yearly_costs = []
        for year in range(years_forward):
            degradation_factor = 1 + (age + year) / \
                building['срок_эксплуатации'] * 0.05
            yearly_costs.append(base_maintenance_cost * degradation_factor)

        return {
            'yearly_costs': yearly_costs,
            'total_cost': sum(yearly_costs),
            'average_cost': sum(yearly_costs) / len(yearly_costs)
        }

    def analyze_building(self, series_name: str, params: Dict) -> Dict:
        """Enhanced building analysis with additional metrics"""
        if series_name not in self.SERIES_DATA:
            raise ValueError(f"Серия {series_name} не найдена")

        building_data = self.SERIES_DATA[series_name]
        survey_data = self.generate_survey(series_name)

        # Calculate basic metrics
        age, remaining_lifetime = self.calculate_building_age(series_name)
        reconstruction_priority = self.calculate_reconstruction_priority(
            series_name, survey_data)
        maintenance_forecast = self.predict_maintenance_costs(series_name)

        # Calculate satisfaction metrics
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
            survey_data,
            satisfaction_weights,
            current_satisfaction
        )

        base_roi, improvement_scenarios = self._calculate_enhanced_roi(
            economics, building_data, params)

        # Prepare analysis results
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
        """Prepare enhanced base metrics DataFrame with consistent string types"""
        base_metrics = {
            'Показатель': [
                'Годы строительства',
                'Этажность',
                'Подъездов',
                'Площадь застройки',
                'Жилая площадь',
                'Полезная площадь',
                'Число квартир',
                'Возраст здания',
                'Остаточный ресурс',
                'Материал конструкции'
            ],
            'Значение': [
                f"({building_data['годы_строительства'][0]}, {building_data['годы_строительства'][1]})",
                str(building_data['этажность']),  # Convert to string
                str(building_data['подъездов']),  # Convert to string
                f"{building_data['площадь_застройки']} м²",
                f"{building_data['жилая_площадь']} м²",
                f"{building_data['полезная_площадь']} м²",
                str(building_data['число_квартир']),  # Convert to string
                f"{age} лет",
                f"{remaining_lifetime:.1%}",
                building_data['материал']
            ]
        }

        # Add apartment information with string conversion
        for apt_type, count in building_data['квартиры'].items():
            base_metrics['Показатель'].append(f"Квартиры {apt_type}")
            base_metrics['Значение'].append(str(count))  # Convert to string

        return pd.DataFrame(base_metrics)

    def _calculate_enhanced_roi(self, economics: Dict, building_data: Dict, params: Dict) -> Tuple[float, List[Dict]]:
        """Calculate enhanced ROI with additional factors and improvement scenarios"""
        # Базовый расчет ROI
        total_benefit = (
            economics['ожидаемая_стоимость_недвижимости'] -
            building_data['полезная_площадь'] * params['базовая_стоимость_м2'] +
            economics['ежегодная_экономия_на_обслуживании'] * 15
        )

        # Добавляем энергоэффективность
        energy_efficiency_savings = building_data['полезная_площадь'] * 300 * 15
        total_benefit += energy_efficiency_savings

        # Добавляем доход от коммерческих помещений
        commercial_benefit = building_data['полезная_площадь'] * \
            0.1 * 5000 * 12
        total_benefit += commercial_benefit

        base_roi = (total_benefit -
                    economics['общая']) / economics['общая'] * 100

        # Рассчитываем сценарии улучшения
        improvement_scenarios = self.calculate_improvement_scenarios(
            None, building_data)

        return round(base_roi, 2), improvement_scenarios

    def _calculate_payback_period(self, economics: Dict) -> float:
        """Calculate project payback period"""
        annual_benefit = (
            economics['ежегодная_экономия_на_обслуживании'] +
            # примерный годовой доход от повышения стоимости
            economics['ожидаемая_стоимость_недвижимости'] * 0.05
        )
        return round(economics['общая'] / annual_benefit, 1)

    def _estimate_energy_efficiency(self, building_data: Dict, age: int) -> float:
        """Estimate building energy efficiency rating"""
        base_efficiency = 0.7  # базовый коэффициент энергоэффективности
        age_factor = max(0, 1 - age / building_data['срок_эксплуатации'])
        material_factor = 0.9 if building_data['материал'] == 'панель' else 0.8

        efficiency = base_efficiency * age_factor * material_factor
        return round(efficiency, 2)

    def create_satisfaction_plots(self, survey_data: pd.DataFrame):
        """Enhanced visualization with additional insights"""
        # Original box plot
        metrics = {
            'средняя_удовлетворенность_инфраструктурой': 'Инфраструктура',
            'средняя_удовлетворенность_озеленением': 'Озеленение',
            'средняя_удовлетворенность_площадками': 'Площадки',
            'средняя_удовлетворенность_парковкой': 'Парковка',
            'средняя_готовность_к_реконструкции': 'Готовность к реконструкции'
        }

        plot_data = survey_data.melt(
            id_vars=['тип_квартиры', 'количество_жителей'],
            value_vars=list(metrics.keys()),
            var_name='metric',
            value_name='value'
        )

        plot_data['metric'] = plot_data['metric'].map(metrics)

        # Create enhanced box plot
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
            height=600,
            width=1200,
            showlegend=True,
            boxmode='group',
            yaxis_title='Оценка'
        )

        return fig

    def calculate_ipsu(self, survey_data: pd.DataFrame,
                       satisfaction_weights: Dict, current_satisfaction: Dict) -> Tuple[float, float, Dict]:
        """Calculate current and planned IPSU and business metrics"""
        metrics_mapping = {
            'инфраструктура': 'средняя_удовлетворенность_инфраструктурой',
            'озеленение': 'средняя_удовлетворенность_озеленением',
            'площадки': 'средняя_удовлетворенность_площадками',
            'парковка': 'средняя_удовлетворенность_парковкой',
            'техническое_состояние': 'средняя_готовность_к_реконструкции'
        }

        # Calculate current IPSU (simple average of current satisfaction)
        current_values = [current_satisfaction[metric]
                          for metric in metrics_mapping.keys()]
        current_ipsu = sum(current_values) / len(current_values)

        # Calculate planned IPSU (weighted average based on target values)
        weighted_sum = sum(satisfaction_weights.get(f'вес_{metric}', 0)
                           for metric in metrics_mapping.keys())
        total_weight = len(metrics_mapping)

        if total_weight <= 0:
            planned_ipsu = 0.0
        else:
            planned_ipsu = weighted_sum / total_weight

        # Business metrics calculation
        improvement_potential = {
            metric: max(0, satisfaction_weights.get(f'вес_{metric}', 0) -
                        current_satisfaction.get(metric, 0))
            for metric in metrics_mapping.keys()
        }

        valid_metrics = [
            col for col in metrics_mapping.values() if col in survey_data.columns]
        satisfaction_stability = np.mean(
            [survey_data[col].std() for col in valid_metrics]) if valid_metrics else 0

        business_metrics = {
            'potential_satisfaction_increase': sum(improvement_potential.values()) / len(improvement_potential) if improvement_potential else 0,
            'critical_areas': [metric for metric, potential in improvement_potential.items() if potential > 0.3],
            'satisfaction_stability': satisfaction_stability
        }

        return round(current_ipsu, 3), round(planned_ipsu, 3), business_metrics

    def calculate_improvement_scenarios(self, series_name: str, building_data: Dict) -> List[Dict]:
        """Calculate different improvement scenarios for ROI optimization"""
        scenarios = []

        # Сценарий 1: Надстройка этажа
        additional_floor = {
            'название': 'Надстройка этажа',
            'площадь_застройки': building_data['площадь_застройки'],
            'жилая_площадь': building_data['жилая_площадь'] * 1.15,  # +15%
            # +15%
            'полезная_площадь': building_data['полезная_площадь'] * 1.15,
            'число_квартир': building_data['число_квартир'] + (building_data['число_квартир'] / building_data['этажность']),
            'roi_impact': 25  # Примерное влияние на ROI в процентах
        }
        scenarios.append(additional_floor)

        # Сценарий 2: Расширение здания
        expansion = {
            'название': 'Расширение здания',
            # +20%
            'площадь_застройки': building_data['площадь_застройки'] * 1.2,
            'жилая_площадь': building_data['жилая_площадь'] * 1.2,  # +20%
            # +20%
            'полезная_площадь': building_data['полезная_площадь'] * 1.2,
            'число_квартир': building_data['число_квартир'] * 1.2,
            'roi_impact': 30
        }
        scenarios.append(expansion)

        # Сценарий 3: Оптимизация планировок
        optimization = {
            'название': 'Оптимизация планировок',
            'площадь_застройки': building_data['площадь_застройки'],
            'жилая_площадь': building_data['жилая_площадь'] * 1.1,  # +10%
            # +12%
            'полезная_площадь': building_data['полезная_площадь'] * 1.12,
            'число_квартир': building_data['число_квартир'] * 1.15,
            'roi_impact': 20
        }
        scenarios.append(optimization)

        return scenarios

    def _prepare_economics_df(self, economics: Dict) -> pd.DataFrame:
        """Prepare economics DataFrame for display with space as thousand separator"""
        return pd.DataFrame({
            'Показатель': [
                'СМР',
                'Проектирование',
                'Благоустройство',
                'Прочие',
                'Общая стоимость',
                'Ожидаемая стоимость недвижимости',
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
        """Format apartments data for display"""
        return pd.DataFrame({
            'Тип квартиры': list(apartments_dict.keys()),
            'Количество': list(apartments_dict.values())
        })


def run_streamlit_app():
    st.set_page_config(layout="wide")
    st.title('Анализ реконструкции жилых зданий')

    analyzer = BuildingAnalyzer()

    param_to_column = {
        'инфраструктура': 'средняя_удовлетворенность_инфраструктурой',
        'озеленение': 'средняя_удовлетворенность_озеленением',
        'площадки': 'средняя_удовлетворенность_площадками',
        'парковка': 'средняя_удовлетворенность_парковкой',
        'техническое_состояние': 'средняя_готовность_к_реконструкции'
    }

    # Базовая стоимость СМР без улучшений
    BASE_CONSTRUCTION_COST = 55000

    # Коэффициенты влияния целевых значений на стоимость СМР
    COST_IMPACT_COEFFICIENTS = {
        'инфраструктура': 0.3,      # 30% влияния
        'озеленение': 0.15,         # 15% влияния
        'площадки': 0.15,           # 15% влияния
        'парковка': 0.2,            # 20% влияния
        'техническое_состояние': 0.4  # 40% влияния
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

                # Рассчитываем увеличение стоимости на основе разницы между целевым и текущим значением
                improvement = target_value - current_value
                if improvement > 0:
                    # Увеличиваем общее влияние на стоимость с учетом веса параметра
                    total_cost_increase += improvement * \
                        COST_IMPACT_COEFFICIENTS[param]

        # Рассчитываем новую стоимость СМР
        # Максимальное увеличение стоимости - 100% от базовой
        max_increase = 1.0  # 100%
        cost_multiplier = 1 + (total_cost_increase * max_increase)
        adjusted_base_cost = BASE_CONSTRUCTION_COST * cost_multiplier

        # Показываем текущую расчетную стоимость СМР
        st.metric(
            "Расчетная стоимость СМР (за 1м²)",
            f"{int(adjusted_base_cost):,} руб.",
            delta=f"{int(adjusted_base_cost - BASE_CONSTRUCTION_COST):,} руб."
        )

        # Даем возможность скорректировать стоимость вручную
        cost_params = {
            'базовая_стоимость_м2': st.slider(
                'Скорректировать стоимость СМР (за 1м²)',
                40000,
                200000,
                int(adjusted_base_cost)
            ),
        }

    params = {**cost_params, **weights}

    # Остальной код без изменений...
    results = analyzer.analyze_building(selected_series, params)

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
                st.metric(
                    "ИПСУ текущий (на основе опроса)",
                    f"{current_ipsu:.2f}"
                )
            with metrics_col2:
                st.metric(
                    "ИПСУ целевой",
                    f"{planned_ipsu:.2f}",
                    delta=f"{planned_ipsu - current_ipsu:.2f}"
                )

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

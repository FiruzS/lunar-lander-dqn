import gymnasium as gym
import torch
import argparse
from model import DQN
from agent import DQNAgent
import numpy as np


def human_play(env, render=True):
    """Режим игры человеком"""
    print("\n=== Режим: Игра человеком ===")
    print("Управление:")
    print("  0 - Нет действия")
    print("  1 - Двигатель влево")
    print("  2 - Основной двигатель")
    print("  3 - Двигатель вправо")
    print("Нажмите Enter для продолжения...")
    input()
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if render:
            env.render()
        
        # Запрос действия у пользователя
        try:
            action = int(input(f"Введите действие (0-3): "))
            if action < 0 or action > 3:
                print("Неверное действие! Используйте 0-3")
                continue
        except ValueError:
            print("Введите число от 0 до 3")
            continue
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        
        print(f"Награда: {reward:.2f}, Всего: {total_reward:.2f}")
    
    print(f"\nИгра окончена! Общая награда: {total_reward:.2f}")
    return total_reward


def train_dqn(env, num_episodes=500, render=False, save_path="model.pth"):
    """Режим обучения DQN агента"""
    print("\n=== Режим: Обучение DQN ===")
    print(f"Количество эпизодов: {num_episodes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    agent = DQNAgent(
        state_size=8,
        action_size=4,
        hidden_size=128,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update=10,
        memory_size=10000,
        batch_size=64,
        device=device
    )
    
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
            
            # Выбор действия
            action = agent.select_action(state, training=True)
            
            # Выполнение действия
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Сохранение опыта
            agent.memory.push(state, action, reward, next_state, done)
            
            # Оптимизация модели
            loss = agent.optimize_model()
            
            state = next_state
            total_reward += reward
        
        # Затухание epsilon
        agent.decay_epsilon()
        
        # Обновление целевой сети
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        episode_rewards.append(total_reward)
        
        # Статистика
        avg_reward = np.mean(episode_rewards[-100:])
        if episode % 10 == 0:
            print(f"Эпизод {episode+1}/{num_episodes} | "
                  f"Награда: {total_reward:.2f} | "
                  f"Средняя (100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Сохранение лучшей модели
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_model(save_path)
            print(f"✓ Новая лучшая модель! Награда: {best_reward:.2f}")
    
    # Финальное сохранение
    agent.save_model(save_path)
    
    # Построение графика обучения (опционально)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Награда по эпизодам')
        plt.xlabel('Эпизод')
        plt.ylabel('Награда')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        window_size = 100
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg)
            plt.title(f'Скользящее среднее ({window_size} эпизодов)')
            plt.xlabel('Эпизод')
            plt.ylabel('Средняя награда')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300)
        print("\nГрафик обучения сохранен в training_results.png")
    except ImportError:
        print("\nMatplotlib не установлен, пропускаем построение графиков")
    
    return episode_rewards


def computer_play(env, model_path="model.pth", render=True, num_games=5):
    """Режим игры обученного компьютера"""
    print("\n=== Режим: Игра компьютера ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Создание и загрузка агента
    agent = DQNAgent(
        state_size=8,
        action_size=4,
        hidden_size=128,
        device=device
    )
    
    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        print("Сначала обучите модель!")
        return
    
    # Установка epsilon в 0 для жадной стратегии
    agent.epsilon = 0
    
    all_rewards = []
    
    for game in range(num_games):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            if render:
                env.render()
            
            # Выбор лучшего действия (без исследования)
            action = agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
        
        all_rewards.append(total_reward)
        print(f"Игра {game+1}/{num_games}: Награда = {total_reward:.2f}, Шагов = {steps}")
    
    print(f"\n=== Результаты ===")
    print(f"Средняя награда: {np.mean(all_rewards):.2f}")
    print(f"Лучшая награда: {max(all_rewards):.2f}")
    print(f"Худшая награда: {min(all_rewards):.2f}")
    
    return all_rewards


def main():
    parser = argparse.ArgumentParser(description='Lunar Lander - DQN Reinforcement Learning')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['human', 'train', 'computer'],
                       help='Режим работы: human (игра человеком), train (обучение), computer (игра компьютера)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Количество эпизодов для обучения (только для train)')
    parser.add_argument('--model', type=str, default='model.pth',
                       help='Путь к файлу модели (только для computer)')
    parser.add_argument('--games', type=int, default=5,
                       help='Количество игр для компьютера (только для computer)')
    parser.add_argument('--no-render', action='store_true',
                       help='Отключить рендеринг (ускоряет обучение)')
    parser.add_argument('--save', type=str, default='model.pth',
                       help='Путь для сохранения модели')
    
    args = parser.parse_args()
    
    # Создание окружения
    if args.mode == 'human' or args.mode == 'computer':
        env = gym.make('LunarLander-v3', render_mode='human')
    else:
        render_mode = 'human' if not args.no_render else None
        env = gym.make('LunarLander-v3', render_mode=render_mode)
    
    print("=" * 60)
    print("LUNAR LANDER - DQN REINFORCEMENT LEARNING")
    print("=" * 60)
    print(f"\nВыбранный режим: {args.mode.upper()}")
    
    try:
        if args.mode == 'human':
            human_play(env, render=True)
        
        elif args.mode == 'train':
            train_dqn(
                env,
                num_episodes=args.episodes,
                render=not args.no_render,
                save_path=args.save
            )
        
        elif args.mode == 'computer':
            computer_play(
                env,
                model_path=args.model,
                render=True,
                num_games=args.games
            )
    
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    
    finally:
        env.close()
        print("\nРабота завершена")


if __name__ == "__main__":
    main()
